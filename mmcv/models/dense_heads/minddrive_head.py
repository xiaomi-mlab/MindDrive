# ------------------------------------------------------------------------
# Modified from Orion(https://github.com/xiaomi-mlab/Orion)
# Copyright (c) Xiaomi Corporation. All rights reserved.
# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
#  Modified by Shihao Wang
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
from mmcv.models.bricks import Linear
from mmcv.models.utils import bias_init_with_prob

from mmcv.utils import force_fp32
from mmcv.core import build_assigner, build_sampler
from mmcv.core.utils.dist_utils import reduce_mean
from mmcv.core.utils.misc import multi_apply

from mmcv.models.utils import build_transformer,xavier_init
from mmcv.models import HEADS, build_loss
from mmcv.models.dense_heads.anchor_free_head import AnchorFreeHead
from mmcv.models.utils.transformer import inverse_sigmoid

from mmcv.core.bbox import build_bbox_coder
from mmcv.core.bbox.util import normalize_bbox
from mmcv.models.utils import NormedLinear

from mmcv.models.utils.positional_encoding import pos2posemb3d, pos2posemb1d, nerf_positional_encoding
from mmcv.utils.misc import MLN, topk_gather, transform_reference_points, memory_refresh, SELayer_Linear
from mmcv.ops.iou3d_det import nms_gpu
from mmcv.core import xywhr2xyxyr
import os
from mmcv.models.utils.functional import ts2tsemb1d
from mmcv.core.bbox.structures.box_3d_mode import LiDARInstance3DBoxes
from mmcv.models.bricks.transformer import build_transformer_layer_sequence,build_positional_encoding
import numpy as np
from torchvision.transforms.functional import rotate
import torch.nn.functional as F
from mmcv.utils import TORCH_VERSION, digit_version
import copy
from mmcv.models.utils import build_transformer,xavier_init
from mmcv.core.bbox.assigners import *
def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
        
class MLP(nn.Module):
    def __init__(self, in_channels, hidden_unit, verbose=False):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_unit),
            nn.LayerNorm(hidden_unit),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.mlp(x)
        return x

class LaneNet(nn.Module):
    def __init__(self, in_channels, hidden_unit, num_subgraph_layers):
        super(LaneNet, self).__init__()
        self.num_subgraph_layers = num_subgraph_layers
        self.layer_seq = nn.Sequential()
        for i in range(num_subgraph_layers):
            self.layer_seq.add_module(
                f'lmlp_{i}', MLP(in_channels, hidden_unit))
            in_channels = hidden_unit

    def forward(self, pts_lane_feats):
        '''
            Extract lane_feature from vectorized lane representation

        Args:
            pts_lane_feats: [batch size, max_pnum, D]

        Returns:
            inst_lane_feats: [batch size, max_pnum, D]
        '''
        x = pts_lane_feats
        for name, layer in self.layer_seq.named_modules():
            if isinstance(layer, MLP):
                # x [bs, max_lane_num, D]
                x = layer(x)
        return x

@HEADS.register_module()
class MinddriveHead(AnchorFreeHead):
    """Implements the DETR transformer head.
    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.
    Args:
        num_classes (int): Number of categories excluding the background.
        in_channels (int): Number of channels in the input feature map.
        num_query (int): Number of query in Transformer.
        num_reg_fcs (int, optional): Number of fully-connected layers used in
            `FFN`, which is then used for the regression head. Default 2.
        transformer (obj:`mmcv.ConfigDict`|dict): Config for transformer.
            Default: None.
        sync_cls_avg_factor (bool): Whether to sync the avg_factor of
            all ranks. Default to False.
        positional_encoding (obj:`mmcv.ConfigDict`|dict):
            Config for position encoding.
        loss_cls (obj:`mmcv.ConfigDict`|dict): Config of the
            classification loss. Default `CrossEntropyLoss`.
        loss_bbox (obj:`mmcv.ConfigDict`|dict): Config of the
            regression loss. Default `L1Loss`.
        loss_iou (obj:`mmcv.ConfigDict`|dict): Config of the
            regression iou loss. Default `GIoULoss`.
        tran_cfg (obj:`mmcv.ConfigDict`|dict): Training config of
            transformer head.
        test_cfg (obj:`mmcv.ConfigDict`|dict): Testing config of
            transformer head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """
    _version = 2

    def __init__(self,
                 num_classes,
                 in_channels=256,
                 out_dims=4096,
                 embed_dims=256,
                 num_query=100,
                 num_reg_fcs=2,
                 memory_len=1024,
                 topk_proposals=256,
                 num_propagated=256,
                 num_extra=256,
                 n_control=11,
                 can_bus_len=2,
                 with_mask=False,
                 with_dn=True,
                 with_ego_pos=True,
                 match_with_velo=True,
                 match_costs=None,
                 transformer=None,
                 use_memory = False,
                 num_memory = 16,
                 scence_memory_len=256,
                 memory_decoder_transformer = dict(
                    type='CustomTransformerDecoder',
                    num_layers=1,
                    return_intermediate=False,
                    transformerlayers=dict(
                        type='BaseTransformerLayer',
                        attn_cfgs=[
                            dict(
                                type='MultiheadAttention',
                                embed_dims=256,
                                num_heads=8,
                                dropout=0.1),
                        ],
                    feedforward_channels=512,
                    ffn_dropout=0.1,
                    operation_order=('cross_attn', 'norm', 'ffn', 'norm'))),
                 use_col_loss = False,
                 motion_transformer_decoder=dict(
                    type='CustomTransformerDecoder',
                    num_layers=1,
                    return_intermediate=False,
                    transformerlayers=dict(
                        type='BaseTransformerLayer',
                        attn_cfgs=[
                            dict(
                                type='MultiheadAttention',
                                embed_dims=256,
                                num_heads=8,
                                dropout=0.0),
                        ],
                        feedforward_channels=512,
                        ffn_dropout=0.0,
                        operation_order=('cross_attn', 'norm', 'ffn', 'norm'))),
                 motion_map_decoder = None,  
                 with_ego_pose = True, # 默认是打开egopose 和我们以前的ego pose对齐
                 sync_cls_avg_factor=False,
                 code_weights=None,
                 bbox_coder=None,
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     bg_cls_weight=0.1,
                     use_sigmoid=False,
                     loss_weight=1.0,
                     class_weight=1.0),
                 loss_traffic=dict(
                     type='CrossEntropyLoss',
                     bg_cls_weight=0.1,
                     use_sigmoid=False,
                     loss_weight=1.0,
                     class_weight=1.0),
                 loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                 loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                 train_cfg=dict(
                     assigner=dict(
                         type='HungarianAssigner3D',
                         cls_cost=dict(type='ClassificationCost', weight=1.),
                         reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                         iou_cost=dict(
                             type='IoUCost', iou_mode='giou', weight=2.0)),),
                 test_cfg=dict(max_per_img=100),
                 scalar = 5,
                 noise_scale = 0.4,
                 noise_trans = 0.0,
                 dn_weight = 1.0,
                 split = 0.5,
                 init_cfg=None,
                 normedlinear=False,
                 class_agnostic_nms=False,
                 score_threshold=0.,
                 planning_step=None,
                 canbus_dropout=0.0,
                 fut_mode=1,
                 fut_ts = 6,
                 use_pe=True,
                 motion_det_score=None,
                 valid_fut_ts=6,
                 loss_traj=dict(type='L1Loss', loss_weight=0.25),
                 loss_traj_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=0.8),
                 pred_traffic_light_state=False,
                 traffic_light_token=False,
                 state_counter_token=False,
                 state_velo_threshold=0.5,
                 **kwargs):
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since it brings inconvenience when the initialization of
        # `AnchorFreeHead` is called.
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]

        self.code_weights = self.code_weights[:self.code_size]

        if match_costs is not None:
            self.match_costs = match_costs
        else:
            self.match_costs = self.code_weights
        self.with_ego_pose = with_ego_pose
        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor
        class_weight = loss_cls.get('class_weight', None)
        if class_weight is not None and (self.__class__ is MinddriveHead):
            assert isinstance(class_weight, float), 'Expected ' \
                'class_weight to have type float. Found ' \
                f'{type(class_weight)}.'
            # NOTE following the official DETR rep0, bg_cls_weight means
            # relative classification weight of the no-object class.
            bg_cls_weight = loss_cls.get('bg_cls_weight', class_weight)
            assert isinstance(bg_cls_weight, float), 'Expected ' \
                'bg_cls_weight to have type float. Found ' \
                f'{type(bg_cls_weight)}.'
            class_weight = torch.ones(num_classes + 1) * class_weight
            # set background class as the last indice
            class_weight[num_classes] = bg_cls_weight
            loss_cls.update({'class_weight': class_weight})
            if 'bg_cls_weight' in loss_cls:
                loss_cls.pop('bg_cls_weight')
            self.bg_cls_weight = bg_cls_weight

        if train_cfg:
            assert 'assigner' in train_cfg, 'assigner should be provided '\
                'when train_cfg is set.'
            assigner = train_cfg['assigner']


            self.assigner = build_assigner(assigner)
            # DETR sampling=False, so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)

        self.output_dims = out_dims
        self.n_control = n_control
        self.num_query = num_query
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.memory_len = memory_len
        self.topk_proposals = topk_proposals
        self.num_propagated = num_propagated
        self.num_extra = num_extra
        self.with_dn = with_dn
        self.with_ego_pos = with_ego_pos
        self.match_with_velo = match_with_velo
        self.with_mask = with_mask
        self.num_reg_fcs = num_reg_fcs
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.embed_dims = embed_dims
        self.can_bus_len = can_bus_len
        self.use_memory = use_memory
        self.num_memory = num_memory
        self.scence_memory_len = scence_memory_len
        self.scalar = scalar
        self.bbox_noise_scale = noise_scale
        self.bbox_noise_trans = noise_trans
        self.dn_weight = dn_weight
        self.split = split 
        self.fut_mode = fut_mode
        self.motion_transformer_decoder = motion_transformer_decoder
        self.use_col_loss = use_col_loss
        self.use_pe = use_pe
        self.fut_ts = fut_ts
        self.motion_det_score= motion_det_score
        self.act_cfg = transformer.get('act_cfg',
                                       dict(type='ReLU', inplace=True))
        self.num_pred = 6
        self.normedlinear = normedlinear
        if loss_traj_cls['use_sigmoid'] == True:
            self.traj_num_cls = 1
        else:
          self.traj_num_cls = 2

        super(MinddriveHead, self).__init__(num_classes, in_channels, init_cfg = init_cfg)
        
        self.loss_traj = build_loss(loss_traj)
        self.loss_traj_cls = build_loss(loss_traj_cls)
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_iou = build_loss(loss_iou)
        if self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        self.transformer = build_transformer(transformer)

        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights), requires_grad=False)

        self.match_costs = nn.Parameter(torch.tensor(
            self.match_costs), requires_grad=False)

        self.bbox_coder = build_bbox_coder(bbox_coder)

        self.pc_range = nn.Parameter(torch.tensor(
            self.bbox_coder.pc_range), requires_grad=False)

        self.class_agnostic_nms = class_agnostic_nms
        self.score_threshold = score_threshold
        self.planning_step = planning_step
        self.valid_fut_ts = valid_fut_ts

        if self.use_col_loss:
            self.motion_map_decoder = motion_map_decoder
            if self.motion_map_decoder is not None:
                self.lane_encoder = LaneNet(256, 256, 3)
                self.motion_map_decoder = build_transformer(self.motion_map_decoder)
                if self.use_pe:
                    self.pos_mlp = nn.Linear(2, self.embed_dims)
            self.motion_decoder = build_transformer(self.motion_transformer_decoder)
            self.motion_mode_query = nn.Embedding(self.fut_mode, self.embed_dims)	
            self.motion_mode_query.weight.requires_grad = True
            if self.use_pe:
                self.pos_mlp_sa = nn.Linear(2, self.embed_dims)
                
        if self.use_memory:
            self.memory_decoder_transformer = memory_decoder_transformer
            self.memory_query = nn.Embedding(self.num_memory, self.embed_dims)
            self.scene_time_embedding = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.LayerNorm(embed_dims)
            )
            self.memory_decoder_cq = build_transformer(self.memory_decoder_transformer)
            self.memory_decoder_mq = build_transformer(self.memory_decoder_transformer)
        self.canbus_dropout = canbus_dropout
        self.pred_traffic_light_state = pred_traffic_light_state
        if self.pred_traffic_light_state:
            self.loss_traffic = build_loss(loss_traffic)
            # self.loss_affect = build_loss(loss_affect)
        self.traffic_light_token = traffic_light_token
        assert self.pred_traffic_light_state if self.traffic_light_token else True
        self.state_counter_token = state_counter_token
        self.state_velo_threshold = state_velo_threshold

        self._init_layers()
        self.reset_memory()

    def _init_layers(self):
        """Initialize layers of the transformer head."""

        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        if self.normedlinear:
            cls_branch.append(NormedLinear(self.embed_dims, self.cls_out_channels))
        else:
            cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)
        if self.use_col_loss:
            traj_branch = []
            for _ in range(self.num_reg_fcs):
                traj_branch.append(Linear(self.embed_dims*2, self.embed_dims*2))
                traj_branch.append(nn.ReLU())
            traj_branch.append(Linear(self.embed_dims*2, self.fut_ts*2))
            traj_branch = nn.Sequential(*traj_branch)
            motion_num_pred = 1
            self.traj_branches = _get_clones(traj_branch, motion_num_pred)
            self.traj_bg_cls_weight = 0
            traj_cls_branch = []
            for _ in range(self.num_reg_fcs):
                traj_cls_branch.append(Linear(self.embed_dims*2, self.embed_dims*2))
                traj_cls_branch.append(nn.LayerNorm(self.embed_dims*2))
                traj_cls_branch.append(nn.ReLU(inplace=True))
            traj_cls_branch.append(Linear(self.embed_dims*2, self.traj_num_cls))
            traj_cls_branch = nn.Sequential(*traj_cls_branch)
            self.traj_cls_branches = _get_clones(traj_cls_branch, motion_num_pred)

        self.cls_branches = nn.ModuleList(
            [fc_cls for _ in range(self.num_pred)])
        self.reg_branches = nn.ModuleList(
            [reg_branch for _ in range(self.num_pred)])

        self.input_projection = nn.Linear(self.in_channels, self.embed_dims)
        if self.output_dims is not None:
            self.output_projection = nn.Linear(self.embed_dims, self.output_dims)

        self.reference_points = nn.Embedding(self.num_query, 3)
        if self.num_propagated > 0:
            self.pseudo_reference_points = nn.Embedding(self.num_propagated, 3)

        self.query_embedding = nn.Embedding(self.num_extra, self.embed_dims)
        
        if self.output_dims is not None:
            can_bus_layers = [
                    nn.Linear(89, self.embed_dims*4), # canbus + command + egopose (b2d中的can_bus是 18维度)
                    nn.Dropout(p=self.canbus_dropout if hasattr(self,"canbus_dropout") else 0.0),
                    nn.ReLU(),
                    nn.Linear(self.embed_dims*4, self.output_dims)]
            if not hasattr(self,"canbus_dropout") or (hasattr(self,"canbus_dropout") and self.canbus_dropout == 0):
                can_bus_layers.pop(1) # 为了与没有dropout层的早期checkpoints兼容
            self.can_bus_embed = nn.Sequential(*can_bus_layers)
        if hasattr(self,"state_counter_token") and self.state_counter_token:
            state_tick_layers = [
                    nn.Linear(128, self.embed_dims*4), # posemb映射到4096作为输入
                    nn.ReLU(),
                    nn.Linear(self.embed_dims*4, self.output_dims)]
            self.state_tick_emb = nn.Sequential(*state_tick_layers)

        self.query_pos = None

        self.time_embedding = None

        self.ego_pose_pe = None

        if hasattr(self,"planning_step") and self.planning_step is not None:
            self.planning_emb = nn.Sequential(
                nn.Linear((self.planning_step)*3, self.embed_dims*4),
                nn.ReLU(),
                nn.Linear(self.embed_dims*4, self.embed_dims),
            )   
            self.gate_proj = nn.Linear(self.embed_dims, self.embed_dims, bias=False)
            self.down_proj = nn.Linear(self.embed_dims, self.embed_dims, bias=False)
            self.up_proj = nn.Linear(self.embed_dims, self.embed_dims, bias=False)
        
        if hasattr(self,"pred_traffic_light_state") and self.pred_traffic_light_state:
            tl_branch = []
            for _ in range(self.num_reg_fcs):
                tl_branch.append(Linear(self.embed_dims, self.embed_dims))
                tl_branch.append(nn.LayerNorm(self.embed_dims))
                tl_branch.append(nn.ReLU(inplace=True))
            tl_branch.append(Linear(self.embed_dims, 4)) # 3 + 1 , red, yellow, green, affect ego
            fc_tl = nn.Sequential(*tl_branch)
            self.tl_branches = nn.ModuleList(
                [fc_tl for _ in range(self.num_pred)])
            bias_init = bias_init_with_prob(0.01)
            for m in self.tl_branches:
                nn.init.constant_(m[-1].bias, bias_init)
        
        if hasattr(self,"traffic_light_token") and self.traffic_light_token:
            traffic_light_layers = [
                    nn.Linear(self.embed_dims, self.embed_dims*4),
                    nn.ReLU()]
            self.traffic_light_embed = nn.Sequential(*traffic_light_layers)
            self.traffic_light_maxpool = nn.AdaptiveMaxPool1d(1)
            self.traffic_light_lift = nn.Linear(self.embed_dims*4, self.output_dims)

    def init_weights(self):
        """Initialize weights of the transformer head."""
        # The initialization for transformer is important
        nn.init.uniform_(self.reference_points.weight.data, 0, 1)
        if self.num_propagated > 0:
            nn.init.uniform_(self.pseudo_reference_points.weight.data, 0, 1)
            self.pseudo_reference_points.weight.requires_grad = False
        if self.use_memory:
            # 初始化两个decoder
            for p in self.memory_decoder_cq.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            for p in self.memory_decoder_mq.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)
        if self.use_col_loss:
            for p in self.motion_decoder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            nn.init.orthogonal_(self.motion_mode_query.weight)
            if self.use_pe:
                xavier_init(self.pos_mlp_sa, distribution='uniform', bias=0.)
            
            if self.motion_map_decoder is not None:
                for p in self.motion_map_decoder.parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
                for p in self.lane_encoder.parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
                if self.use_pe:
                    xavier_init(self.pos_mlp, distribution='uniform', bias=0.)


    def reset_memory(self):
        self.memory_embedding = None
        self.memory_reference_point = None
        self.memory_timestamp = None
        self.memory_egopose = None
        self.memory_velo = None
        self.sample_time = None
        self.memory_canbus = None
        self.memory_scene_tokens = None
        self.his_memory_canbus_len = None
        if self.state_counter_token:
            self.his_state_counter = None

    def pre_update_memory(self, img_metas, data):
        B = data['img_feats'].size(0)
        # refresh the memory when the scene changes
        if self.memory_embedding is None:
            self.memory_embedding = data['img_feats'].new_zeros(B, self.memory_len, self.embed_dims)
            self.memory_reference_point = data['img_feats'].new_zeros(B, self.memory_len, 3)
            self.memory_timestamp = data['img_feats'].new_zeros(B, self.memory_len, 1)
            self.memory_egopose = data['img_feats'].new_zeros(B, self.memory_len, 4, 4)
            self.memory_velo = data['img_feats'].new_zeros(B, self.memory_len, 2)
            self.sample_time = data['timestamp'].new_zeros(B)
            if self.state_counter_token:
                self.his_state_counter = data['timestamp'].new_zeros(B).to(torch.float)
            self.memory_canbus = data['img_feats'].new_zeros(B, self.can_bus_len, 19)
            self.his_memory_canbus_len = data['timestamp'].new_zeros(B).to(torch.int64)
            x = self.sample_time.to(data['img_feats'].dtype)
            self.memory_scene_tokens = ['' for meta in img_metas]
            if self.use_memory:
                self.memory_scene_query = data['img_feats'].new_zeros(B, self.scence_memory_len, self.embed_dims)
                self.scene_memory_timestamp = data['img_feats'].new_zeros(B, self.scence_memory_len, 1)      
        else:
            self.memory_timestamp += data['timestamp'].unsqueeze(-1).unsqueeze(-1)
            self.sample_time += data['timestamp']
            x = (torch.abs(self.sample_time) < 2.0)
            y = [meta['scene_token'] == memory_tokens for meta, memory_tokens in zip(img_metas, self.memory_scene_tokens)]
            y = torch.tensor(y,device=x.device)
            x = torch.logical_and(x,y).to(data['img_feats'].dtype)
            if self.use_memory:
                self.scene_memory_timestamp += data['timestamp'].unsqueeze(-1).unsqueeze(-1)
                self.scene_memory_timestamp = memory_refresh(self.scene_memory_timestamp[:, :self.scence_memory_len], x)
            self.memory_egopose = data['ego_pose_inv'].unsqueeze(1) @ self.memory_egopose
            self.memory_reference_point = transform_reference_points(self.memory_reference_point, data['ego_pose_inv'], reverse=False)
            self.memory_timestamp = memory_refresh(self.memory_timestamp[:, :self.memory_len], x)
            self.memory_reference_point = memory_refresh(self.memory_reference_point[:, :self.memory_len], x)
            self.memory_embedding = memory_refresh(self.memory_embedding[:, :self.memory_len], x)
            self.memory_egopose = memory_refresh(self.memory_egopose[:, :self.memory_len], x)
            self.memory_velo = memory_refresh(self.memory_velo[:, :self.memory_len], x)
            for i, his_len in enumerate(self.his_memory_canbus_len):
                self.memory_canbus[i:i+1, :his_len.to(torch.int64), 1:4] -= data['can_bus'][i:i+1, :3].unsqueeze(1)
                self.memory_canbus[i:i+1, :his_len.to(torch.int64), -1] -= data['can_bus'][i:i+1, -1].unsqueeze(1)
            self.memory_canbus = memory_refresh(self.memory_canbus[:, :self.can_bus_len], x)
            self.his_memory_canbus_len = memory_refresh(self.his_memory_canbus_len, x)
            if self.state_counter_token:
                self.his_state_counter = memory_refresh(self.his_state_counter, x)
            self.sample_time = data['timestamp'].new_zeros(B)
            if self.use_memory:
                self.memory_scene_query = memory_refresh(self.memory_scene_query[:, :self.scence_memory_len], x) 
        # for the first frame, padding pseudo_reference_points (non-learnable)
        if self.num_propagated > 0:
            pseudo_reference_points = self.pseudo_reference_points.weight * (self.pc_range[3:6] - self.pc_range[0:3]) + self.pc_range[0:3]
            self.memory_reference_point[:, :self.num_propagated]  = self.memory_reference_point[:, :self.num_propagated] + (1 - x).view(B, 1, 1) * pseudo_reference_points
            self.memory_egopose[:, :self.num_propagated]  = self.memory_egopose[:, :self.num_propagated] + (1 - x).view(B, 1, 1, 1) * torch.eye(4, device=x.device)

    def post_update_memory(self, img_metas, data, rec_ego_pose, all_cls_scores, all_bbox_preds, outs_dec, mask_dict, rec_can_bus, history_query=None):
        if self.training and mask_dict and mask_dict['pad_size'] > 0:
            rec_reference_points = all_bbox_preds[:, :, mask_dict['pad_size']:, :3][-1]
            rec_velo = all_bbox_preds[:, :, mask_dict['pad_size']:, -2:][-1]
            out_memory = outs_dec[:, :, mask_dict['pad_size']:, :][-1]
            rec_score = all_cls_scores[:, :, mask_dict['pad_size']:, :][-1].sigmoid().topk(1, dim=-1).values[..., 0:1]
            rec_timestamp = torch.zeros_like(rec_score, dtype=torch.float64)
        else:
            rec_reference_points = all_bbox_preds[..., :3][-1]
            rec_velo = all_bbox_preds[..., -2:][-1]
            out_memory = outs_dec[-1]
            rec_score = all_cls_scores[-1].sigmoid().topk(1, dim=-1).values[..., 0:1]
            rec_timestamp = torch.zeros_like(rec_score, dtype=torch.float64)
        
        # topk proposals
        _, topk_indexes = torch.topk(rec_score, self.topk_proposals, dim=1)
        rec_timestamp = topk_gather(rec_timestamp, topk_indexes)
        rec_reference_points = topk_gather(rec_reference_points, topk_indexes).detach()
        rec_memory = topk_gather(out_memory, topk_indexes).detach()
        rec_ego_pose = topk_gather(rec_ego_pose, topk_indexes)
        rec_velo = topk_gather(rec_velo, topk_indexes).detach()
        self.memory_embedding = torch.cat([rec_memory, self.memory_embedding], dim=1)
        self.memory_timestamp = torch.cat([rec_timestamp, self.memory_timestamp], dim=1)
        if self.use_memory:
            self.scene_memory_timestamp = torch.cat([torch.zeros_like(self.scene_memory_timestamp[:, :self.num_memory,:], dtype=torch.float64), self.scene_memory_timestamp], dim=1)
            self.scene_memory_timestamp -= data['timestamp'].unsqueeze(-1).unsqueeze(-1)
        self.memory_egopose= torch.cat([rec_ego_pose, self.memory_egopose], dim=1)
        self.memory_reference_point = torch.cat([rec_reference_points, self.memory_reference_point], dim=1)
        self.memory_velo = torch.cat([rec_velo, self.memory_velo], dim=1)
        self.memory_canbus = torch.cat([rec_can_bus, self.memory_canbus], dim=1)
        self.his_memory_canbus_len += 1 
        self.memory_reference_point = transform_reference_points(self.memory_reference_point, data['ego_pose'], reverse=False)
        self.memory_timestamp -= data['timestamp'].unsqueeze(-1).unsqueeze(-1)
        self.sample_time -= data['timestamp']
        self.memory_egopose = data['ego_pose'].unsqueeze(1) @ self.memory_egopose
        for i, his_len in enumerate(self.his_memory_canbus_len):
            self.memory_canbus[i:i+1, :his_len.to(torch.int64), 1:4] += data['can_bus'][i:i+1, :3].unsqueeze(1)
            self.memory_canbus[i:i+1, :his_len.to(torch.int64), -1] += data['can_bus'][i:i+1, -1].unsqueeze(1)
        self.memory_scene_tokens = [meta['scene_token'] for meta in img_metas]
        if self.use_memory:
            self.memory_scene_query = torch.cat([history_query.detach(), self.memory_scene_query], dim=1)
        return out_memory

    def temporal_alignment(self, query_pos, tgt, reference_points):
        B = query_pos.size(0)

        temp_reference_point = (self.memory_reference_point - self.pc_range[:3]) / (self.pc_range[3:6] - self.pc_range[0:3])
        temp_pos = self.query_pos(nerf_positional_encoding(temp_reference_point.repeat(1, 1, self.n_control))) 
        temp_memory = self.memory_embedding
        rec_ego_pose = torch.eye(4, device=query_pos.device).unsqueeze(0).unsqueeze(0).repeat(B, query_pos.size(1), 1, 1)
        
        if self.with_ego_pos:
            rec_ego_motion = torch.cat([torch.zeros_like(reference_points[...,:1]), rec_ego_pose[..., :3, :].flatten(-2)], dim=-1)
            rec_ego_motion = nerf_positional_encoding(rec_ego_motion)
            memory_ego_motion = torch.cat([self.memory_timestamp, self.memory_egopose[..., :3, :].flatten(-2)], dim=-1).float()
            memory_ego_motion = nerf_positional_encoding(memory_ego_motion)
            temp_pos = self.ego_pose_pe(temp_pos, memory_ego_motion)

        query_pos += self.time_embedding(pos2posemb1d(torch.zeros_like(reference_points[...,:1])))
        temp_pos += self.time_embedding(pos2posemb1d(self.memory_timestamp).float())

        if self.num_propagated > 0:
            tgt = torch.cat([tgt, temp_memory[:, :self.num_propagated]], dim=1)
            query_pos = torch.cat([query_pos, temp_pos[:, :self.num_propagated]], dim=1)
            reference_points = torch.cat([reference_points, temp_reference_point[:, :self.num_propagated]], dim=1)
            rec_ego_pose = torch.eye(4, device=query_pos.device).unsqueeze(0).unsqueeze(0).repeat(B, query_pos.shape[1]+self.num_propagated, 1, 1)
            temp_memory = temp_memory[:, self.num_propagated:]
            temp_pos = temp_pos[:, self.num_propagated:]
            
        return tgt, query_pos, reference_points, temp_memory, temp_pos, rec_ego_pose

    def prepare_for_dn(self, batch_size, reference_points, img_metas):
        if self.training and self.with_dn:
            targets = [torch.cat((img_meta['gt_bboxes_3d']._data.gravity_center, img_meta['gt_bboxes_3d']._data.tensor[:, 3:]),dim=1) for img_meta in img_metas ]
            labels = [img_meta['gt_labels_3d']._data for img_meta in img_metas ]
            known = [(torch.ones_like(t)).cuda() for t in labels]
            know_idx = known
            unmask_bbox = unmask_label = torch.cat(known)
            #gt_num
            known_num = [t.size(0) for t in targets]
        
            labels = torch.cat([t for t in labels])
            boxes = torch.cat([t for t in targets])
            batch_idx = torch.cat([torch.full((t.size(0), ), i) for i, t in enumerate(targets)])
        
            known_indice = torch.nonzero(unmask_label + unmask_bbox)
            known_indice = known_indice.view(-1)
            # add noise
            known_indice = known_indice.repeat(self.scalar, 1).view(-1)
            known_labels = labels.repeat(self.scalar, 1).view(-1).long().to(reference_points.device)
            known_bid = batch_idx.repeat(self.scalar, 1).view(-1)
            known_bboxs = boxes.repeat(self.scalar, 1).to(reference_points.device)
            if self.pred_traffic_light_state:
                traffic_state_mask = [img_meta['traffic_state_mask']._data for img_meta in img_metas]
                traffic_state = [img_meta['traffic_state']._data for img_meta in img_metas]
                traffic_state_mask  = torch.cat([t for t in traffic_state_mask])
                traffic_state = torch.cat([t for t in traffic_state])
                known_traffic_state_mask = traffic_state_mask.repeat(self.scalar, 1).view(-1).long().to(reference_points.device)
                known_traffic_state = traffic_state.repeat(self.scalar, 1).long().to(reference_points.device)
                pass #TODO @DIANKUN
            known_bbox_center = known_bboxs[:, :3].clone()
            known_bbox_scale = known_bboxs[:, 3:6].clone()

            if self.bbox_noise_scale > 0:
                diff = known_bbox_scale / 2 + self.bbox_noise_trans
                rand_prob = torch.rand_like(known_bbox_center) * 2 - 1.0
                known_bbox_center += torch.mul(rand_prob,
                                            diff) * self.bbox_noise_scale
                known_bbox_center[..., 0:3] = (known_bbox_center[..., 0:3] - self.pc_range[0:3]) / (self.pc_range[3:6] - self.pc_range[0:3])

                known_bbox_center = known_bbox_center.clamp(min=0.0, max=1.0)
                mask = torch.norm(rand_prob, 2, 1) > self.split
                known_labels[mask] = self.num_classes
            
            single_pad = int(max(known_num))
            pad_size = int(single_pad * self.scalar)
            padding_bbox = torch.zeros(pad_size, 3).to(reference_points.device)
            padded_reference_points = torch.cat([padding_bbox, reference_points], dim=0).unsqueeze(0).repeat(batch_size, 1, 1)

            if len(known_num):
                map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
                map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(self.scalar)]).long()
            if len(known_bid):
                padded_reference_points[(known_bid.long(), map_known_indice)] = known_bbox_center.to(reference_points.device)

            tgt_size = pad_size + self.num_query + self.num_extra
            attn_mask = torch.ones(tgt_size, tgt_size).to(reference_points.device) < 0
            # match query cannot see the reconstruct
            attn_mask[pad_size:, :pad_size] = True
            # reconstruct cannot see each other
            for i in range(self.scalar):
                if i == 0:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                if i == self.scalar - 1:
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
                else:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
             
            # update dn mask for temporal modeling
            query_size = pad_size + self.num_query + self.num_propagated + self.num_extra
            tgt_size = pad_size + self.num_query + self.memory_len + self.num_extra
            temporal_attn_mask = torch.ones(query_size, tgt_size).to(reference_points.device) < 0
            temporal_attn_mask[:attn_mask.size(0), :attn_mask.size(1)] = attn_mask 
            temporal_attn_mask[pad_size:, :pad_size] = True
            if self.with_mask:
                temporal_attn_mask[pad_size+self.num_extra:, pad_size:pad_size+self.num_extra] = True
            attn_mask = temporal_attn_mask

            mask_dict = {
                'known_indice': torch.as_tensor(known_indice).long(),
                'batch_idx': torch.as_tensor(batch_idx).long(),
                'map_known_indice': torch.as_tensor(map_known_indice).long(),
                'known_lbs_bboxes': (known_labels, known_bboxs),
                'know_idx': know_idx,
                'pad_size': pad_size
            }
            if self.pred_traffic_light_state:
                mask_dict.update(dict(known_lbs_bboxes=(known_labels, known_bboxs, known_traffic_state, known_traffic_state_mask)))
            
        else:
            attn_mask = None
            if self.with_mask:
                tgt_size = self.num_query + self.memory_len + self.num_extra
                query_size = self.num_query + self.num_propagated + self.num_extra
                attn_mask = torch.ones(query_size, tgt_size).to(reference_points.device) < 0
                attn_mask[self.num_extra:, :self.num_extra] = True
            padded_reference_points = reference_points.unsqueeze(0).repeat(batch_size, 1, 1)
            mask_dict = None

        return padded_reference_points, attn_mask, mask_dict

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """load checkpoints."""
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since `AnchorFreeHead._load_from_state_dict` should not be
        # called here. Invoking the default `Module._load_from_state_dict`
        # is enough.

        # Names of some parameters in has been changed.
        version = local_metadata.get('version', None)
        if (version is None or version < 2) and self.__class__ is MinddriveHead:
            convert_dict = {
                '.self_attn.': '.attentions.0.',
                # '.ffn.': '.ffns.0.',
                '.multihead_attn.': '.attentions.1.',
                '.decoder.norm.': '.decoder.post_norm.'
            }
            state_dict_keys = list(state_dict.keys())
            for k in state_dict_keys:
                for ori_key, convert_key in convert_dict.items():
                    if ori_key in k:
                        convert_key = k.replace(ori_key, convert_key)
                        state_dict[convert_key] = state_dict[k]
                        del state_dict[k]

        super(AnchorFreeHead,
              self)._load_from_state_dict(state_dict, prefix, local_metadata,
                                          strict, missing_keys,
                                          unexpected_keys, error_msgs)
    
    def forward(self, img_metas, pos_embed, map_metas, **data):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """
        # zero init the memory bank
        self.pre_update_memory(img_metas, data)
        x = data['img_feats']
        B, N, C, H, W = x.shape
        num_tokens = N * H * W
        memory = x.permute(0, 1, 3, 4, 2).reshape(B, num_tokens, C)

        memory = self.input_projection(memory)

        reference_points = self.reference_points.weight
        reference_points = torch.cat([torch.zeros_like(reference_points[:self.num_extra]), reference_points], dim=0)
        reference_points, attn_mask, mask_dict = self.prepare_for_dn(B, reference_points, img_metas)
        query_pos = self.query_pos(nerf_positional_encoding(reference_points.repeat(1, 1, self.n_control)))
        tgt = torch.zeros_like(query_pos)

        # prepare for the tgt and query_pos using mln.
        tgt, query_pos, reference_points, temp_memory, temp_pos, rec_ego_pose = self.temporal_alignment(query_pos, tgt, reference_points)
        
        if self.use_memory :    
            current_query = self.memory_query.weight.unsqueeze(0).repeat(B,1,1) # (4, 16, 256)
            temp_scene_query = self.memory_scene_query # (4, 256, 256)
            # time embeding
            time_embeding = self.scene_time_embedding(pos2posemb1d(self.scene_memory_timestamp).float()) # (4, 256, 1)  -> (4, 256, 256) 
            cur_time_embeding = torch.zeros_like(current_query) # (4, 16, 256) 
            # scene query <-> memory scene query
            temp_query_embedding = self.memory_decoder_mq(
                query=current_query,
                key=temp_scene_query,
                # value=temp_scene_query.permute(1, 0, 2), # (256, bs, 256)
                query_pos=cur_time_embeding,
                key_pos=time_embeding,
                # key_padding_mask=None
                )
        
        if mask_dict and mask_dict['pad_size'] > 0:
            tgt[:, mask_dict['pad_size']:mask_dict['pad_size']+self.num_extra, :] = self.query_embedding.weight.unsqueeze(0)
            query_pos[:, mask_dict['pad_size']:mask_dict['pad_size']+self.num_extra, :] = query_pos[:, mask_dict['pad_size']:mask_dict['pad_size']+self.num_extra, :] * 0
        else:
            tgt[:, :self.num_extra, :] = self.query_embedding.weight.unsqueeze(0)
            query_pos[:, :self.num_extra, :] = query_pos[:, :self.num_extra, :] * 0
            

        # transformer here is a little different from PETR
        outs_dec = self.transformer(tgt, memory, query_pos, pos_embed, attn_mask, temp_memory, temp_pos)
        if mask_dict and mask_dict['pad_size'] > 0:
            reference_points = torch.cat([reference_points[:, :mask_dict['pad_size'], :], reference_points[:, mask_dict['pad_size']+self.num_extra:, :]], dim=-2)
        else:
            reference_points = reference_points[:, self.num_extra:, :]

        outs_dec = torch.nan_to_num(outs_dec)
        if mask_dict and mask_dict['pad_size'] > 0: # mask_dict['pad_size']: 100, self.num_extra: 256
            vlm_memory = outs_dec[-1, :, mask_dict['pad_size']:mask_dict['pad_size']+self.num_extra, :] # (6, 1, 1256, 256) -> (256, 256)
            outs_dec = torch.cat([outs_dec[:, :, :mask_dict['pad_size'], :], outs_dec[:, :, mask_dict['pad_size']+self.num_extra:, :]], dim=-2) # (6, 1, 1256, 256) -> # (6, 1, 1000, 256)
        else:
            vlm_memory = outs_dec[-1, :, :self.num_extra, :]
            outs_dec = outs_dec[:, :, self.num_extra:, :]
        if self.use_memory :
            # scene query <-> scene query
            history_query = self.memory_decoder_cq(
                query=temp_query_embedding,
                key=vlm_memory,
                query_pos=None,
                key_pos=None,
                )
        
        outputs_classes = []
        outputs_coords = []
        outputs_coords_bev =[]
        if self.use_col_loss:
            outputs_trajs = []
            outputs_trajs_classes = []
        outputs_traffic_states = []
        for lvl in range(outs_dec.shape[0]):
            reference = inverse_sigmoid(reference_points.clone())
            assert reference.shape[-1] == 3
            outputs_class = self.cls_branches[lvl](outs_dec[lvl])
            if self.pred_traffic_light_state:
                outputs_traffic_state = self.tl_branches[lvl](outs_dec[lvl])
                outputs_traffic_states.append(outputs_traffic_state)
            tmp = self.reg_branches[lvl](outs_dec[lvl])

            tmp[..., 0:3] += reference[..., 0:3]
            tmp[..., 0:3] = tmp[..., 0:3].sigmoid()
            
            outputs_coords_bev.append(tmp[..., 0:2].clone().detach())

            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        if self.pred_traffic_light_state:
            all_traffic_states = torch.stack(outputs_traffic_states)
        all_cls_scores = torch.stack(outputs_classes)
        all_bbox_preds = torch.stack(outputs_coords)
        all_bbox_preds[..., 0:3] = (all_bbox_preds[..., 0:3] * (self.pc_range[3:6] - self.pc_range[0:3]) + self.pc_range[0:3])

        rec_can_bus = data['can_bus'].clone()
        rec_can_bus[:, :3] = 0
        rec_can_bus[:, -1] = 0
        rec_can_bus = torch.cat([data['command'].unsqueeze(-1), rec_can_bus], dim=-1) # (1, 1+18)
        memory_ego_pose = self.memory_egopose.reshape(B, -1, self.topk_proposals, 4, 4).flatten(-2)
        if self.output_dims is not None:
            if self.use_memory:
                vlm_memory = torch.cat((vlm_memory, history_query), dim=1)
            vlm_memory = self.output_projection(vlm_memory) # (B, 256, 256) -> (B, 256, 4096)
            can_bus_input = torch.cat([rec_can_bus, self.memory_canbus.flatten(-2), memory_ego_pose.mean(-2).flatten(-2)], dim=-1) # (1, 19+19*2+16*2)
            can_bus_input = can_bus_input.to(torch.float32)
            can_bus_embed = self.can_bus_embed(can_bus_input) # (1, 89) -> (1, 4096)
            if self.with_ego_pose: 
                vlm_memory = torch.cat([vlm_memory, can_bus_embed.unsqueeze(-2)], dim=-2) # (1, 257, 4096)
            else:
                vlm_memory = vlm_memory

            if self.traffic_light_token:
                if mask_dict and mask_dict['pad_size'] > 0:
                    traffic_light_cls_score = all_cls_scores[-1, :, mask_dict['pad_size']:, 6]
                    det_embed = outs_dec[-1, :, mask_dict['pad_size']:, :]
                else:
                    traffic_light_cls_score = all_cls_scores[-1,:,:,6]
                    det_embed = outs_dec[-1, :, :, :]

                logits, topk_index = torch.topk(traffic_light_cls_score,dim=-1,k=30)
                traffic_feature = topk_gather(det_embed, topk_index)
                traffic_embed = self.traffic_light_embed(traffic_feature)
                traffic_embed = self.traffic_light_maxpool(traffic_embed.permute(0,2,1)).permute(0,2,1)
                traffic_embed = self.traffic_light_lift(traffic_embed)
                vlm_memory = torch.cat([vlm_memory, traffic_embed], dim=-2) # (1, 258, 4096)

            if self.state_counter_token:
                memory_timestamp = self.memory_timestamp[:,0:self.memory_len:self.num_propagated]
                # memory_ego_trans_xy = self.memory_canbus[:,:,1:3]
                memory_ego_velo_xy = torch.linalg.norm(torch.cat([rec_can_bus[:,8:10].unsqueeze(1), self.memory_canbus[:,:,8:10]],dim=1),dim=-1)
                state_counter_mask = torch.logical_and(memory_ego_velo_xy.mean(dim=-1)<self.state_velo_threshold,(self.his_memory_canbus_len >= self.can_bus_len)) #过去三帧的平均速度小于阈值（这里暂定为1）
                if state_counter_mask.any():
                    ts_interval = memory_timestamp[state_counter_mask][:,0,:].max()
                    self.his_state_counter[state_counter_mask] += ts_interval
                self.his_state_counter[~state_counter_mask] = 0.
                state_counter_emb = ts2tsemb1d(self.his_state_counter.unsqueeze(-1))   
                state_counter_emb = self.state_tick_emb(state_counter_emb)
                vlm_memory = torch.cat([vlm_memory, state_counter_emb.unsqueeze(-2)], dim=-2) # (1, 258, 4096)
                pass
            
        if self.use_col_loss:
            batch_size, num_agent = outputs_coords_bev[-1].shape[:2]
            motion_query = outs_dec[-1].permute(1, 0, 2) 
            mode_query = self.motion_mode_query.weight
            motion_query = (motion_query[:, None, :, :] + mode_query[None, :, None, :]).flatten(0, 1)
            if self.use_pe:
                motion_coords = outputs_coords_bev[-1]
                motion_pos = self.pos_mlp_sa(motion_coords)
                motion_pos = motion_pos.unsqueeze(2).repeat(1, 1, self.fut_mode, 1).flatten(1, 2)
            else:
                motion_pos = None

            if self.motion_det_score is not None:
                motion_score = outputs_classes[-1]
                max_motion_score = motion_score.max(dim=-1)[0]
                invalid_motion_idx = max_motion_score < self.motion_det_score  # [B, A]
                invalid_motion_idx = invalid_motion_idx.unsqueeze(2).repeat(1, 1, self.fut_mode).flatten(1, 2)
            else:
                invalid_motion_idx = None

            motion_query = motion_query.permute(1, 0, 2)
            motion_hs = self.motion_decoder(
                query=motion_query,
                key=motion_query,
                query_pos=motion_pos,
                key_pos=motion_pos,
                attn_mask=invalid_motion_idx
                )
            if self.motion_map_decoder is not None:
                self.map_num_vec = 11
                self.map_num = 300
                motion_coords = outputs_coords_bev[-1]  # [B, A, 2]
                motion_coords = motion_coords.unsqueeze(2).repeat(1, 1, self.fut_mode, 1).flatten(1, 2)
                map_query = map_metas['outs_dec_one2one'][-1].view(batch_size, self.map_num, -1)

                map_query = self.lane_encoder(map_query)  # [B, P, pts, D] -> [B, P, D]
                map_score = map_metas['all_lane_cls_one2one'][-1]
                map_pos = map_metas['all_lane_preds_one2one'][-1].reshape(B, self.map_num, self.map_num_vec, 3)
                map_pos = map_pos[...,:2]
                map_query, map_pos, key_padding_mask = self.select_and_pad_pred_map(
                    motion_coords, map_query, map_score, map_pos,
                    map_thresh=0.5, dis_thresh=0.2,
                    pe_normalization=True, use_fix_pad=True)
                # map_query = map_query.permute(1, 0, 2)  # [P, B*M, D]  batch first 不用转变 [B*M, P, D]
                ca_motion_query = motion_hs.permute(1, 0, 2).flatten(0, 1).unsqueeze(0)
                ca_motion_query = ca_motion_query.permute(1, 0, 2)

                if self.use_pe:
                    (num_query, batch) = ca_motion_query.shape[:2] 
                    motion_pos = torch.zeros((num_query, batch, 2), device=motion_hs.device)
                    motion_pos = self.pos_mlp(motion_pos)
                    map_pos = map_pos.permute(1, 0, 2)
                    map_pos = self.pos_mlp(map_pos)
                else:
                    motion_pos, map_pos = None, None
                
                ca_motion_query = self.motion_map_decoder(
                    query=ca_motion_query, # [1, 26880, 256]
                    key=map_query,  # [111, 26880, 128])
                    # value=map_query,
                    query_pos=motion_pos, # [1, 26880, 256]
                    key_pos=map_pos.permute(1, 0, 2), # [111, 26880, 256]
                    key_padding_mask=key_padding_mask) # [26880, 111]
            else:
                ca_motion_query = motion_hs.flatten(0, 1).unsqueeze(0)
            motion_hs = motion_hs.unflatten(
                dim=1, sizes=(num_agent, self.fut_mode)
            )
            ca_motion_query = ca_motion_query.squeeze(0).unflatten(
                dim=0, sizes=(batch_size, num_agent, self.fut_mode)
            ).squeeze(3)
            motion_hs = torch.cat([motion_hs, ca_motion_query], dim=-1)  # [B, A, fut_mode, 2D]
        

            outputs_traj = self.traj_branches[0](motion_hs)
            outputs_trajs.append(outputs_traj)
            outputs_traj_class = self.traj_cls_branches[0](motion_hs)
            outputs_trajs_classes.append(outputs_traj_class.squeeze(-1))
            (batch, num_agent) = motion_hs.shape[:2]
            outputs_trajs = torch.stack(outputs_trajs)
            outputs_trajs_classes = torch.stack(outputs_trajs_classes)

            
        if self.use_memory:
            # update the memory bank
            out_memory = self.post_update_memory(img_metas, data, rec_ego_pose, all_cls_scores, all_bbox_preds, outs_dec, mask_dict, rec_can_bus.unsqueeze(-2), history_query)
        else:
            out_memory = self.post_update_memory(img_metas, data, rec_ego_pose, all_cls_scores, all_bbox_preds, outs_dec, mask_dict, rec_can_bus.unsqueeze(-2))
        if mask_dict and mask_dict['pad_size'] > 0:
            output_known_class = all_cls_scores[:, :, :mask_dict['pad_size'], :]
            output_known_coord = all_bbox_preds[:, :, :mask_dict['pad_size'], :]
            # output_known_trajs = outputs_trajs[:, :, :mask_dict['pad_size'], :]
            # output_known_trajs_classes = outputs_trajs_classes[:, :, :mask_dict['pad_size'], :]
            outputs_class = all_cls_scores[:, :, mask_dict['pad_size']:, :]
            outputs_coord = all_bbox_preds[:, :, mask_dict['pad_size']:, :]
            if self.use_col_loss:
                outputs_trajs = outputs_trajs[:, :, mask_dict['pad_size']:, :]
                outputs_trajs_classes = outputs_trajs_classes[:, :, mask_dict['pad_size']:, :]
                mask_dict['output_known_lbs_bboxes']=(output_known_class, output_known_coord)
                outs = {
                    'all_cls_scores': outputs_class,
                    'all_bbox_preds': outputs_coord,
                    'dn_mask_dict':mask_dict,
                    'all_traj_preds': outputs_trajs.repeat(outputs_coord.shape[0], 1, 1, 1, 1),
                    'all_traj_cls_scores': outputs_trajs_classes.repeat(outputs_coord.shape[0], 1, 1, 1),
                }
            else:
                mask_dict['output_known_lbs_bboxes']=(output_known_class, output_known_coord)
                outs = {
                    'all_cls_scores': outputs_class,
                    'all_bbox_preds': outputs_coord,
                    'dn_mask_dict':mask_dict,
                }
            if self.pred_traffic_light_state:
                output_known_traffic_states = all_traffic_states[:, :, :mask_dict['pad_size'], :]
                outputs_traffic_states = all_traffic_states[:, :, mask_dict['pad_size']:, :]
                mask_dict['output_known_lbs_bboxes']=(output_known_class, output_known_coord, output_known_traffic_states)
                outs.update(dict(dn_mask_dict = mask_dict))
                outs.update(dict(all_traffic_states = outputs_traffic_states))
        else:
            if self.use_col_loss:
                outs = {
                    'all_cls_scores': all_cls_scores,
                    'all_bbox_preds': all_bbox_preds,
                    'dn_mask_dict':None,
                    'all_traj_preds': outputs_trajs.repeat(all_bbox_preds.shape[0], 1, 1, 1, 1),
                    'all_traj_cls_scores': outputs_trajs_classes.repeat(all_bbox_preds.shape[0], 1, 1, 1),
                    }
            else:   
                outs = {
                    'all_cls_scores': all_cls_scores,
                    'all_bbox_preds': all_bbox_preds,
                    'dn_mask_dict':None,
                }
            if self.pred_traffic_light_state:
                outs.update(dict(all_traffic_states = all_traffic_states))

        return outs, vlm_memory
    
    def prepare_for_loss(self, mask_dict):
        """
        prepare dn components to calculate loss
        Args:
            mask_dict: a dict that contains dn information
        """
        if self.pred_traffic_light_state:
            output_known_class, output_known_coord, output_known_traffic_state = mask_dict['output_known_lbs_bboxes']
            known_labels, known_bboxs, known_traffic_state, known_traffic_state_mask = mask_dict['known_lbs_bboxes']
        else:
            output_known_class, output_known_coord = mask_dict['output_known_lbs_bboxes']
            known_labels, known_bboxs = mask_dict['known_lbs_bboxes']
        map_known_indice = mask_dict['map_known_indice'].long()
        known_indice = mask_dict['known_indice'].long().cpu()
        batch_idx = mask_dict['batch_idx'].long()
        bid = batch_idx[known_indice]
        if len(output_known_class) > 0:
            output_known_class = output_known_class.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)
            output_known_coord = output_known_coord.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)
            if self.pred_traffic_light_state:
                output_known_traffic_state = output_known_traffic_state.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)
        num_tgt = known_indice.numel()
        if self.pred_traffic_light_state:
            return known_labels, known_bboxs, known_traffic_state, known_traffic_state_mask, output_known_class, output_known_coord, num_tgt, output_known_traffic_state
        else:
            return known_labels, known_bboxs, output_known_class, output_known_coord, num_tgt


    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_attr_labels,
                           traffic_pred=None, 
                           gt_traffic_state=None, 
                           gt_traffic_mask=None,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indexes for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indexes for each image.
                - neg_inds (Tensor): Sampled negative indexes for each image.
        """

        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        if self.use_col_loss:
            gt_fut_trajs = gt_attr_labels[:, :self.fut_ts*2]
            gt_fut_masks = gt_attr_labels[:, self.fut_ts*2:self.fut_ts*3]
            num_gt_bbox, gt_traj_c = gt_fut_trajs.shape

        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                                gt_labels, gt_bboxes_ignore, self.match_costs, self.match_with_velo)
        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes, ),
                                    self.num_classes,
                                    dtype=torch.long)
        label_weights = gt_bboxes.new_ones(num_bboxes)
        label_traffic_state = gt_bboxes.new_zeros(num_bboxes,2).to(torch.long)
        label_traffic_mask =  gt_bboxes.new_zeros(num_bboxes).to(torch.bool)
        # bbox targets
        code_size = gt_bboxes.size(1)
        bbox_targets = torch.zeros_like(bbox_pred)[..., :code_size]
        bbox_weights = torch.zeros_like(bbox_pred)
        # print(gt_bboxes.size(), bbox_pred.size())
        if self.use_col_loss:
            # trajs targets
            traj_targets = torch.zeros((num_bboxes, gt_traj_c), dtype=torch.float32, device=bbox_pred.device)
            traj_weights = torch.zeros_like(traj_targets)
            traj_targets[pos_inds] = gt_fut_trajs[sampling_result.pos_assigned_gt_inds]
            traj_weights[pos_inds] = 1.0

            # Filter out invalid fut trajs
            traj_masks = torch.zeros_like(traj_targets)  # [num_bboxes, fut_ts*2]
            gt_fut_masks = gt_fut_masks.unsqueeze(-1).repeat(1, 1, 2).view(num_gt_bbox, -1)  # [num_gt_bbox, fut_ts*2]
            traj_masks[pos_inds] = gt_fut_masks[sampling_result.pos_assigned_gt_inds]
            traj_weights = traj_weights * traj_masks

            # Extra future timestamp mask for controlling pred horizon
            fut_ts_mask = torch.zeros((num_bboxes, self.fut_ts, 2),
                                    dtype=torch.float32, device=bbox_pred.device)
            fut_ts_mask[:, :self.valid_fut_ts, :] = 1.0
            fut_ts_mask = fut_ts_mask.view(num_bboxes, -1)
            traj_weights = traj_weights * fut_ts_mask

        # DETR
        if sampling_result.num_gts > 0:
            bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
            bbox_weights[pos_inds] = 1.0
            labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
            if gt_traffic_state is not None:
                label_traffic_state[pos_inds] = gt_traffic_state[sampling_result.pos_assigned_gt_inds]
                label_traffic_mask[pos_inds] = gt_traffic_mask[sampling_result.pos_assigned_gt_inds]
        if self.use_col_loss:
            return (labels, label_weights, bbox_targets, label_traffic_state, label_traffic_mask, bbox_weights, traj_targets,
                    traj_weights, traj_masks.view(-1, self.fut_ts, 2)[..., 0],
                    pos_inds, neg_inds)
        else:        
            return (labels, label_weights, bbox_targets, label_traffic_state, label_traffic_mask, bbox_weights, pos_inds, neg_inds)

    def select_and_pad_pred_map(
            self,
            motion_pos,
            map_query,
            map_score,
            map_pos,
            map_thresh=0.5,
            dis_thresh=None,
            pe_normalization=True,
            use_fix_pad=False
        ):
            """select_and_pad_pred_map.
            Args:
                motion_pos: [B, A, 2]
                map_query: [B, P, D].
                map_score: [B, P, 3].
                map_pos: [B, P, pts, 2].
                map_thresh: map confidence threshold for filtering low-confidence preds
                dis_thresh: distance threshold for masking far maps for each agent in cross-attn
                use_fix_pad: always pad one lane instance for each batch
            Returns:
                selected_map_query: [B*A, P1(+1), D], P1 is the max inst num after filter and pad.
                selected_map_pos: [B*A, P1(+1), 2]
                selected_padding_mask: [B*A, P1(+1)]
            """
            
            if dis_thresh is None:
                raise NotImplementedError('Not implement yet')

            # use the most close pts pos in each map inst as the inst's pos
            batch, num_map = map_pos.shape[:2]
            map_dis = torch.sqrt(map_pos[..., 0]**2 + map_pos[..., 1]**2)
            min_map_pos_idx = map_dis.argmin(dim=-1).flatten()  # [B*P]
            min_map_pos = map_pos.flatten(0, 1)  # [B*P, pts, 2]
            min_map_pos = min_map_pos[range(min_map_pos.shape[0]), min_map_pos_idx]  # [B*P, 2]
            min_map_pos = min_map_pos.view(batch, num_map, 2)  # [B, P, 2]

            # select & pad map vectors for different batch using map_thresh
            map_score = map_score.sigmoid()
            map_max_score = map_score.max(dim=-1)[0]
            map_idx = map_max_score > map_thresh
            batch_max_pnum = 0
            for i in range(map_score.shape[0]):
                pnum = map_idx[i].sum()
                if pnum > batch_max_pnum:
                    batch_max_pnum = pnum

            selected_map_query, selected_map_pos, selected_padding_mask = [], [], []
            for i in range(map_score.shape[0]):
                dim = map_query.shape[-1]
                valid_pnum = map_idx[i].sum()
                valid_map_query = map_query[i, map_idx[i]]
                valid_map_pos = min_map_pos[i, map_idx[i]]
                pad_pnum = batch_max_pnum - valid_pnum
                padding_mask = torch.tensor([False], device=map_score.device).repeat(batch_max_pnum)
                if pad_pnum != 0:
                    valid_map_query = torch.cat([valid_map_query, torch.zeros((pad_pnum, dim), device=map_score.device)], dim=0)
                    valid_map_pos = torch.cat([valid_map_pos, torch.zeros((pad_pnum, 2), device=map_score.device)], dim=0)
                    padding_mask[valid_pnum:] = True
                selected_map_query.append(valid_map_query)
                selected_map_pos.append(valid_map_pos)
                selected_padding_mask.append(padding_mask)

            selected_map_query = torch.stack(selected_map_query, dim=0)
            selected_map_pos = torch.stack(selected_map_pos, dim=0)
            selected_padding_mask = torch.stack(selected_padding_mask, dim=0)

            # generate different pe for map vectors for each agent
            num_agent = motion_pos.shape[1]
            selected_map_query = selected_map_query.unsqueeze(1).repeat(1, num_agent, 1, 1)  # [B, A, max_P, D]
            selected_map_pos = selected_map_pos.unsqueeze(1).repeat(1, num_agent, 1, 1)  # [B, A, max_P, 2]
            selected_padding_mask = selected_padding_mask.unsqueeze(1).repeat(1, num_agent, 1)  # [B, A, max_P]
            # move lane to per-car coords system
            selected_map_dist = selected_map_pos - motion_pos[:, :, None, :]  # [B, A, max_P, 2]
            if pe_normalization:
                selected_map_pos = selected_map_pos - motion_pos[:, :, None, :]  # [B, A, max_P, 2]

            # filter far map inst for each agent
            map_dis = torch.sqrt(selected_map_dist[..., 0]**2 + selected_map_dist[..., 1]**2)
            valid_map_inst = (map_dis <= dis_thresh)  # [B, A, max_P]
            invalid_map_inst = (valid_map_inst == False)
            selected_padding_mask = selected_padding_mask + invalid_map_inst

            selected_map_query = selected_map_query.flatten(0, 1)
            selected_map_pos = selected_map_pos.flatten(0, 1)
            selected_padding_mask = selected_padding_mask.flatten(0, 1)

            num_batch = selected_padding_mask.shape[0]
            feat_dim = selected_map_query.shape[-1]
            if use_fix_pad:
                pad_map_query = torch.zeros((num_batch, 1, feat_dim), device=selected_map_query.device)
                pad_map_pos = torch.ones((num_batch, 1, 2), device=selected_map_pos.device)
                pad_lane_mask = torch.tensor([False], device=selected_padding_mask.device).unsqueeze(0).repeat(num_batch, 1)
                selected_map_query = torch.cat([selected_map_query, pad_map_query], dim=1)
                selected_map_pos = torch.cat([selected_map_pos, pad_map_pos], dim=1)
                selected_padding_mask = torch.cat([selected_padding_mask, pad_lane_mask], dim=1)

            return selected_map_query, selected_map_pos, selected_padding_mask



    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_attr_labels_list,
                    traffic_pred_list, 
                    gt_traffic_state_list, 
                    gt_traffic_mask_list,
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indexes for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]
        if gt_traffic_state_list is None:
            traffic_pred_list = [None for _ in cls_scores_list]
            gt_traffic_state_list = [None for _ in cls_scores_list]
            gt_traffic_mask_list = [None for _ in cls_scores_list]
        if gt_attr_labels_list is None:
            gt_attr_labels_list = [None for _ in cls_scores_list]
        if self.use_col_loss:
            (labels_list, label_weights_list, bbox_targets_list, traffic_targets_list, traffic_mask_list,
            bbox_weights_list, traj_targets_list, traj_weights_list,
            gt_fut_masks_list, pos_inds_list, neg_inds_list) = multi_apply(
                self._get_target_single, cls_scores_list, bbox_preds_list,
                gt_labels_list, gt_bboxes_list, gt_attr_labels_list, 
                traffic_pred_list, gt_traffic_state_list, gt_traffic_mask_list, 
                gt_bboxes_ignore_list)
            num_total_pos = sum((inds.numel() for inds in pos_inds_list))
            num_total_neg = sum((inds.numel() for inds in neg_inds_list))
            return (labels_list, label_weights_list, bbox_targets_list, traffic_targets_list, traffic_mask_list, bbox_weights_list,
                    traj_targets_list, traj_weights_list, gt_fut_masks_list, num_total_pos, num_total_neg)
        else:
            (labels_list, label_weights_list, bbox_targets_list, traffic_targets_list, traffic_mask_list,
            bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
                self._get_target_single, cls_scores_list, bbox_preds_list,
                gt_labels_list, gt_bboxes_list, gt_attr_labels_list,
                traffic_pred_list, gt_traffic_state_list, gt_traffic_mask_list, 
                gt_bboxes_ignore_list)
            num_total_pos = sum((inds.numel() for inds in pos_inds_list))
            num_total_neg = sum((inds.numel() for inds in neg_inds_list))
            return (labels_list, label_weights_list, bbox_targets_list, traffic_targets_list, traffic_mask_list,
                    bbox_weights_list, num_total_pos, num_total_neg)

    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    traj_preds=None,
                    traj_cls_preds = None,
                    gt_attr_labels_list = None,
                    traffic_preds=None, 
                    gt_traffic_state_list=None,
                    gt_traffic_mask_list=None,
                    gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indexes for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        traffic_pred_list = [traffic_preds[i] if traffic_preds is not None else None for i in range(num_imgs) ]

        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                        gt_bboxes_list, gt_labels_list, gt_attr_labels_list,
                                        traffic_pred_list, gt_traffic_state_list, gt_traffic_mask_list,
                                        gt_bboxes_ignore_list)
        if self.use_col_loss:
            (labels_list, label_weights_list, bbox_targets_list, traffic_targets_list, traffic_mask_list, bbox_weights_list,
            traj_targets_list, traj_weights_list, gt_fut_masks_list,
            num_total_pos, num_total_neg) = cls_reg_targets
        else:  
            (labels_list, label_weights_list, bbox_targets_list, traffic_targets_list, traffic_mask_list, bbox_weights_list,
            num_total_pos, num_total_neg) = cls_reg_targets

        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        if self.use_col_loss:
            traj_targets = torch.cat(traj_targets_list, 0)
            traj_weights = torch.cat(traj_weights_list, 0)
            gt_fut_masks = torch.cat(gt_fut_masks_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)


        loss_traffic = torch.tensor(0.,device=loss_cls.device)
        loss_affect = torch.tensor(0.,device=loss_cls.device)
        if self.pred_traffic_light_state:
            traffic_state_label = torch.cat(traffic_targets_list)
            traffic_state_mask = torch.cat(traffic_mask_list)
            traffic_state_pred = traffic_preds.reshape(-1, traffic_preds.shape[-1])
            traffic_light_label = traffic_state_label[:,0]
            traffic_affect_target = traffic_state_label[:,1]
            traffic_light_pred = traffic_state_pred[:,:3]
            traffic_affect_pred = traffic_state_pred[:,3:]
            traffic_avg_factor = traffic_state_mask.sum()
            if traffic_state_mask.sum():
                loss_traffic = self.loss_traffic(traffic_light_pred[traffic_state_mask.to(torch.bool)], 
                                    traffic_light_label[traffic_state_mask.to(torch.bool)].to(torch.long), avg_factor=traffic_avg_factor) # multi-class
                loss_affect = self.loss_traffic(traffic_affect_pred[traffic_state_mask.to(torch.bool)], 
                                    torch.logical_not(traffic_affect_target[traffic_state_mask.to(torch.bool)]).to(torch.long))  # single-class
                loss_traffic = torch.nan_to_num(loss_traffic)
                loss_affect = torch.nan_to_num(loss_affect)
            else:
                loss_traffic = traffic_light_pred.sum() * 0. # dummy loss 
                loss_affect = traffic_light_pred.sum() * 0.

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
                bbox_preds[isnotnan, :10], normalized_bbox_targets[isnotnan, :10], bbox_weights[isnotnan, :10], avg_factor=num_total_pos)

        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)
        if self.use_col_loss:
            # traj regression loss
            best_traj_preds = self.get_best_fut_preds(
                traj_preds.reshape(-1, self.fut_mode, self.fut_ts, 2),
                traj_targets.reshape(-1, self.fut_ts, 2), gt_fut_masks)

            neg_inds = (bbox_weights[:, 0] == 0)
            traj_labels = self.get_traj_cls_target(
                traj_preds.reshape(-1, self.fut_mode, self.fut_ts, 2),
                traj_targets.reshape(-1, self.fut_ts, 2),
                gt_fut_masks, neg_inds)

            loss_traj = self.loss_traj(
                best_traj_preds[isnotnan],
                traj_targets[isnotnan],
                traj_weights[isnotnan],
                avg_factor=num_total_pos)

            # if self.use_traj_lr_warmup:
            if False:
                loss_scale_factor = get_traj_warmup_loss_weight(self.epoch, self.tot_epoch)
                loss_traj = loss_scale_factor * loss_traj

            # traj classification loss
            traj_cls_scores = traj_cls_preds.reshape(-1, self.fut_mode)
            # construct weighted avg_factor to match with the official DETR repo
            traj_cls_avg_factor = num_total_pos * 1.0 + \
                num_total_neg * self.traj_bg_cls_weight
            if self.sync_cls_avg_factor:
                traj_cls_avg_factor = reduce_mean(
                    traj_cls_scores.new_tensor([traj_cls_avg_factor]))

            traj_cls_avg_factor = max(traj_cls_avg_factor, 1)
            loss_traj_cls = self.loss_traj_cls(
                traj_cls_scores, traj_labels, label_weights, avg_factor=traj_cls_avg_factor
            )

            if digit_version(TORCH_VERSION) >= digit_version('1.8'):
                loss_traj = torch.nan_to_num(loss_traj)
                loss_traj_cls = torch.nan_to_num(loss_traj_cls)
        else:
            loss_traj = torch.zeros_like(loss_cls)
            loss_traj_cls = torch.zeros_like(loss_cls)

        return loss_cls, loss_bbox, loss_traffic, loss_affect, loss_traj, loss_traj_cls
   
    def dn_loss_single(self,
                    cls_scores,
                    bbox_preds,
                    traffic_scores,
                    known_bboxs,
                    known_labels,
                    known_traffic_state, 
                    known_traffic_state_mask,
                    num_total_pos=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indexes for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 3.14159 / 6 * self.split * self.split  * self.split ### positive rate
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        bbox_weights = torch.ones_like(bbox_preds)
        label_weights = torch.ones_like(known_labels)
        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, known_labels.long(), label_weights, avg_factor=cls_avg_factor)

        loss_traffic = torch.tensor(0.,device=loss_cls.device)
        loss_affect = torch.tensor(0.,device=loss_cls.device)
        if traffic_scores is not None:
            assert known_traffic_state is not None and known_traffic_state_mask is not None
            traffic_avg_factor = known_traffic_state_mask.sum()
            if traffic_avg_factor:
                loss_traffic = self.loss_traffic(traffic_scores[known_traffic_state_mask.to(torch.bool)][:,:-1], 
                                    known_traffic_state[known_traffic_state_mask.to(torch.bool)][:,0].to(torch.long), avg_factor=traffic_avg_factor)
                loss_affect = self.loss_traffic(traffic_scores[known_traffic_state_mask.to(torch.bool)][:,-1:], 
                                    torch.logical_not(known_traffic_state[known_traffic_state_mask.to(torch.bool)][:,1]).to(torch.long), avg_factor=traffic_avg_factor)
            else:
                loss_traffic = traffic_scores.sum() * 0. # dummy loss 
                loss_affect = traffic_scores.sum() * 0.
        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(known_bboxs, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)

        bbox_weights = bbox_weights * self.code_weights

        
        loss_bbox = self.loss_bbox(
                bbox_preds[isnotnan, :10], normalized_bbox_targets[isnotnan, :10], bbox_weights[isnotnan, :10], avg_factor=num_total_pos)

        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)
        loss_traffic = torch.nan_to_num(loss_traffic)
        loss_affect = torch.nan_to_num(loss_affect)
        
        return self.dn_weight * loss_cls, self.dn_weight * loss_bbox, self.dn_weight * loss_traffic, self.dn_weight * loss_affect
    
    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             preds_dicts,
             gt_attr_labels,
             gt_traffic_state=None,
             gt_traffic_state_mask=None,
             gt_bboxes_ignore=None):
        """"Loss function.
        Args:
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indexes for each
                image with shape (num_gts, ).
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device
        gt_bboxes_list = [torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
            dim=1).to(device) for gt_bboxes in gt_bboxes_list]

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]
        if self.pred_traffic_light_state:
            all_traffic_preds = preds_dicts['all_traffic_states']
            all_gt_traffic_state = [gt_traffic_state for _ in range(num_dec_layers)]
            all_gt_traffic_mask = [gt_traffic_state_mask for _ in range(num_dec_layers)]
        else:
            all_traffic_preds = [None for _ in range(num_dec_layers)]
            all_gt_traffic_state = [None for _ in range(num_dec_layers)]
            all_gt_traffic_mask = [None for _ in range(num_dec_layers)]
        if self.use_col_loss:
            all_traj_preds = preds_dicts['all_traj_preds'] # ([6, 4, 1120, 6, 12])
            all_traj_cls_scores = preds_dicts['all_traj_cls_scores']# ([6, 4, 1120, 6])
            all_gt_attr_labels_list = [gt_attr_labels for _ in range(num_dec_layers)]
        else:
            all_traj_preds = [None for _ in range(num_dec_layers)]
            all_traj_cls_scores = [None for _ in range(num_dec_layers)]
            all_gt_attr_labels_list = [None for _ in range(num_dec_layers)] 

        losses_cls, losses_bbox, losses_traffic, losses_affect, loss_traj, loss_traj_cls= multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds,
            all_gt_bboxes_list, all_gt_labels_list, all_traj_preds,
            all_traj_cls_scores,all_gt_attr_labels_list,
            all_traffic_preds, all_gt_traffic_state, all_gt_traffic_mask,
            all_gt_bboxes_ignore_list)


        

        loss_dict = dict()

        # loss_dict['size_loss'] = size_loss
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        if self.use_col_loss:
            loss_dict['loss_traj'] = loss_traj[-1]
            loss_dict['loss_traj_cls'] = loss_traj_cls[-1]

            # Planning Loss
            batch, num_agent = all_traj_preds[-1].shape[:2]
            agent_fut_preds = all_traj_preds[-1].view(batch, num_agent, self.fut_mode, self.fut_ts, 2)
            agent_fut_cls_preds = all_traj_cls_scores[-1].view(batch, num_agent, self.fut_mode)
        if self.pred_traffic_light_state:
            loss_dict['loss_traffic'] = losses_traffic[-1]
            loss_dict['loss_affect'] = losses_affect[-1]


        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_traffic_i, loss_affect_i in zip(losses_cls[:-1],
                                           losses_bbox[:-1],
                                           losses_traffic[:-1],
                                           losses_affect[:-1],
                                           ):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            if self.pred_traffic_light_state:
                loss_dict[f'd{num_dec_layer}.loss_traffic'] = loss_traffic_i
                loss_dict[f'd{num_dec_layer}.loss_affect'] = loss_affect_i
            num_dec_layer += 1
        
        if preds_dicts['dn_mask_dict'] is not None:
            if self.pred_traffic_light_state:
                known_labels, known_bboxs, known_traffic_state, known_traffic_state_mask, output_known_class, output_known_coord, num_tgt, output_known_traffic_state = self.prepare_for_loss(preds_dicts['dn_mask_dict'])
            else:
                known_labels, known_bboxs, output_known_class, output_known_coord, num_tgt = self.prepare_for_loss(preds_dicts['dn_mask_dict'])
            all_known_bboxs_list = [known_bboxs for _ in range(num_dec_layers)]
            all_known_labels_list = [known_labels for _ in range(num_dec_layers)]
            all_num_tgts_list = [
                num_tgt for _ in range(num_dec_layers)
            ]
            if self.pred_traffic_light_state:
                all_known_traffic_state_list = [known_traffic_state for _ in range(num_dec_layers)]
                all_known_traffic_state_mask_list = [known_traffic_state_mask for _ in range(num_dec_layers)]
            else:
                all_known_traffic_state_list = [None for _ in range(num_dec_layers)]
                all_known_traffic_state_mask_list = [None for _ in range(num_dec_layers)]
                output_known_traffic_state = [None for _ in range(num_dec_layers)]
            dn_losses_cls, dn_losses_bbox, dn_losses_traffic, dn_losses_affect = multi_apply(
                self.dn_loss_single, output_known_class, output_known_coord, output_known_traffic_state,
                all_known_bboxs_list, all_known_labels_list, all_known_traffic_state_list, all_known_traffic_state_mask_list,
                all_num_tgts_list)
            loss_dict['dn_loss_cls'] = dn_losses_cls[-1]
            loss_dict['dn_loss_bbox'] = dn_losses_bbox[-1]
            if self.pred_traffic_light_state:
                loss_dict['dn_loss_traffic'] = dn_losses_traffic[-1]
                loss_dict['dn_loss_affect'] = dn_losses_affect[-1]
            num_dec_layer = 0
            for loss_cls_i, loss_bbox_i, loss_traffic_i, loss_affect_i in zip(dn_losses_cls[:-1],
                                            dn_losses_bbox[:-1],
                                            dn_losses_traffic[:-1],
                                            dn_losses_affect[:-1]):
                loss_dict[f'd{num_dec_layer}.dn_loss_cls'] = loss_cls_i
                loss_dict[f'd{num_dec_layer}.dn_loss_bbox'] = loss_bbox_i
                if self.pred_traffic_light_state:
                    loss_dict[f'd{num_dec_layer}.dn_loss_traffic'] = loss_traffic_i
                    loss_dict[f'd{num_dec_layer}.dn_loss_affect'] = loss_affect_i
                num_dec_layer += 1
                
        elif self.with_dn:
            dn_losses_cls, dn_losses_bbox = multi_apply(
                self.loss_single, all_cls_scores, all_bbox_preds,
                all_gt_bboxes_list, all_gt_labels_list, 
                all_gt_bboxes_ignore_list)
            loss_dict['dn_loss_cls'] = dn_losses_cls[-1].detach()
            loss_dict['dn_loss_bbox'] = dn_losses_bbox[-1].detach()     
            num_dec_layer = 0
            for loss_cls_i, loss_bbox_i in zip(dn_losses_cls[:-1],
                                            dn_losses_bbox[:-1]):
                loss_dict[f'd{num_dec_layer}.dn_loss_cls'] = loss_cls_i.detach()     
                loss_dict[f'd{num_dec_layer}.dn_loss_bbox'] = loss_bbox_i.detach()     
                num_dec_layer += 1
        if self.use_col_loss:
            agent_outs = {
                    'agent_preds': all_bbox_preds[-1][..., 0:2],
                    'agent_fut_preds': agent_fut_preds,
                    'agent_score_preds':all_cls_scores[-1].sigmoid(),
                    'agent_fut_cls_preds': agent_fut_cls_preds.sigmoid(),
                }
            return loss_dict, agent_outs
        else:
            return loss_dict

    @force_fp32(apply_to=('preds_dicts'))
    def get_motion_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        preds_dicts = self.bbox_coder.decode(preds_dicts)
        num_samples = len(preds_dicts)

        if os.getenv('DEBUG_SHOW_PRED', None) is not None:
            score_threshold = self.score_threshold
        else:
            score_threshold = 0.
        if isinstance(score_threshold, list):
            assert len(score_threshold) == self.num_classes, \
                "score_threshold length must = class_names, len class_names: {}".format(self.num_classes)
        elif isinstance(score_threshold, dict):
            for dist_range, cls_scores_thr in score_threshold.items():
                assert len(cls_scores_thr) == self.num_classes, \
                "dist_range ---> score_threshold length must = class_names, class_names: {}".format(self.num_classes)
        else:
            score_threshold = [score_threshold] * self.num_classes

        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            # bboxes = img_metas[i]['box_type_3d'](bboxes, bboxes.size(-1))
            bboxes = LiDARInstance3DBoxes(bboxes, bboxes.size(-1))
            scores = preds['scores']
            labels = preds['labels']
            trajs = preds['trajs']

            cur_bboxes = []
            cur_labels = []
            cur_scores = []
            cur_trajs = []
            for cid in range(self.num_classes):
                cid_mask = labels == cid
                score_thrs = scores.new_ones(scores.shape) * score_threshold[cid]
                score_mask = scores > score_thrs
                mask = cid_mask & score_mask
                cur_bboxes.append(bboxes[mask].tensor)
                cur_labels.append(labels[mask])
                cur_scores.append(scores[mask])
                cur_trajs.append(trajs[mask])

            bboxes = torch.cat(cur_bboxes)
            scores = torch.cat(cur_scores)
            labels = torch.cat(cur_labels)
            trajs =  torch.cat(cur_trajs)
            bboxes = LiDARInstance3DBoxes(bboxes, bboxes.size(-1))

            if self.class_agnostic_nms and os.getenv('DEBUG_SHOW_DIR_PRED', None) is not None:
                score_weight = torch.zeros_like(labels).float()
                if isinstance(self.class_agnostic_nms, dict):
                    class_list = self.class_agnostic_nms['classes']
                    compensate = self.class_agnostic_nms['compensate']
                    class_idxes = list()
                    mask = torch.zeros_like(labels).bool()
                    for i, l in enumerate(labels):
                        for tgt_label, comp in zip(class_list, compensate):
                            if l == tgt_label:
                                score_weight[i] = comp
                                break
                    for l in class_list:
                        mask = torch.bitwise_or(mask, labels == l)
                    class_idxes = torch.where(mask)[0]
                    keep_idxes = torch.where(torch.bitwise_not(mask))[0]
                else:
                    class_idxes = torch.arange(start=0, end=labels.size()[0], step=1, device=labels.device).long()
                    keep_idxes = None
                if class_idxes.size()[0]:
                    boxes_for_nms = xywhr2xyxyr(bboxes.bev[class_idxes])
                    scores_for_nms = scores[class_idxes] + score_weight
                    # the nms in 3d detection just remove overlap boxes.
                    nms_thr = self.class_agnostic_nms['nms_thr']
                    if isinstance(nms_thr, (list, tuple)):
                        min_score, max_score = nms_thr
                        nms_thr = np.random.rand()*(max_score - min_score) + min_score
                    selected = nms_gpu(
                        boxes_for_nms.cuda(),
                        scores_for_nms.cuda(),
                        thresh=nms_thr,
                        pre_maxsize=self.class_agnostic_nms['pre_max_size'],
                        post_max_size=self.class_agnostic_nms['post_max_size'])
                    if keep_idxes is not None:
                        selected = torch.cat([class_idxes[selected], keep_idxes], 0)
                    else:
                        selected = class_idxes[selected]
                    bboxes = bboxes[selected]
                    scores = scores[selected]
                    labels = labels[selected]
                    trajs = trajs[selected]

            ret_list.append([bboxes, scores, labels, trajs])
        return ret_list

    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        preds_dicts = self.bbox_coder.decode(preds_dicts)
        num_samples = len(preds_dicts)

        if os.getenv('DEBUG_SHOW_PRED', None) is not None:
            score_threshold = self.score_threshold
        else:
            score_threshold = 0.
        if isinstance(score_threshold, list):
            assert len(score_threshold) == self.num_classes, \
                "score_threshold length must = class_names, len class_names: {}".format(self.num_classes)
        elif isinstance(score_threshold, dict):
            for dist_range, cls_scores_thr in score_threshold.items():
                assert len(cls_scores_thr) == self.num_classes, \
                "dist_range ---> score_threshold length must = class_names, class_names: {}".format(self.num_classes)
        else:
            score_threshold = [score_threshold] * self.num_classes

        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            bboxes = LiDARInstance3DBoxes(bboxes, bboxes.size(-1))
            scores = preds['scores']
            labels = preds['labels']

            cur_bboxes = []
            cur_labels = []
            cur_scores = []
            for cid in range(self.num_classes):
                cid_mask = labels == cid
                score_thrs = scores.new_ones(scores.shape) * score_threshold[cid]
                score_mask = scores > score_thrs
                mask = cid_mask & score_mask
                cur_bboxes.append(bboxes[mask].tensor)
                cur_labels.append(labels[mask])
                cur_scores.append(scores[mask])

            bboxes = torch.cat(cur_bboxes)
            scores = torch.cat(cur_scores)
            labels = torch.cat(cur_labels)
            # bboxes = img_metas[i]['box_type_3d'](bboxes, bboxes.size(-1))
            bboxes = LiDARInstance3DBoxes(bboxes, bboxes.size(-1))
            if self.class_agnostic_nms and os.getenv('DEBUG_SHOW_DIR_PRED', None) is not None:
                score_weight = torch.zeros_like(labels).float()
                if isinstance(self.class_agnostic_nms, dict):
                    class_list = self.class_agnostic_nms['classes']
                    compensate = self.class_agnostic_nms['compensate']
                    class_idxes = list()
                    mask = torch.zeros_like(labels).bool()
                    for i, l in enumerate(labels):
                        for tgt_label, comp in zip(class_list, compensate):
                            if l == tgt_label:
                                score_weight[i] = comp
                                break
                    for l in class_list:
                        mask = torch.bitwise_or(mask, labels == l)
                    class_idxes = torch.where(mask)[0]
                    keep_idxes = torch.where(torch.bitwise_not(mask))[0]
                else:
                    class_idxes = torch.arange(start=0, end=labels.size()[0], step=1, device=labels.device).long()
                    keep_idxes = None
                if class_idxes.size()[0]:
                    boxes_for_nms = xywhr2xyxyr(bboxes.bev[class_idxes])
                    scores_for_nms = scores[class_idxes] + score_weight
                    # the nms in 3d detection just remove overlap boxes.
                    nms_thr = self.class_agnostic_nms['nms_thr']
                    if isinstance(nms_thr, (list, tuple)):
                        min_score, max_score = nms_thr
                        nms_thr = np.random.rand()*(max_score - min_score) + min_score
                    selected = nms_gpu(
                        boxes_for_nms.cuda(),
                        scores_for_nms.cuda(),
                        thresh=nms_thr,
                        pre_maxsize=self.class_agnostic_nms['pre_max_size'],
                        post_max_size=self.class_agnostic_nms['post_max_size'])
                    if keep_idxes is not None:
                        selected = torch.cat([class_idxes[selected], keep_idxes], 0)
                    else:
                        selected = class_idxes[selected]
                    bboxes = bboxes[selected]
                    scores = scores[selected]
                    labels = labels[selected]

            ret_list.append([bboxes, scores, labels])
        return ret_list

    def get_best_fut_preds(self,
             traj_preds,
             traj_targets,
             gt_fut_masks):
        """"Choose best preds among all modes.
        Args:
            traj_preds (Tensor): MultiModal traj preds with shape (num_box_preds, fut_mode, fut_ts, 2).
            traj_targets (Tensor): Ground truth traj for each pred box with shape (num_box_preds, fut_ts, 2).
            gt_fut_masks (Tensor): Ground truth traj mask with shape (num_box_preds, fut_ts).
            pred_box_centers (Tensor): Pred box centers with shape (num_box_preds, 2).
            gt_box_centers (Tensor): Ground truth box centers with shape (num_box_preds, 2).

        Returns:
            best_traj_preds (Tensor): best traj preds (min displacement error with gt)
                with shape (num_box_preds, fut_ts*2).
        """

        cum_traj_preds = traj_preds.cumsum(dim=-2)
        cum_traj_targets = traj_targets.cumsum(dim=-2)

        # Get min pred mode indices.
        # (num_box_preds, fut_mode, fut_ts)
        dist = torch.linalg.norm(cum_traj_targets[:, None, :, :] - cum_traj_preds, dim=-1)
        dist = dist * gt_fut_masks[:, None, :]
        dist = dist[..., -1]
        dist[torch.isnan(dist)] = dist[torch.isnan(dist)] * 0
        min_mode_idxs = torch.argmin(dist, dim=-1).tolist()
        box_idxs = torch.arange(traj_preds.shape[0]).tolist()
        best_traj_preds = traj_preds[box_idxs, min_mode_idxs, :, :].reshape(-1, self.fut_ts*2)

        return best_traj_preds

    def get_traj_cls_target(self,
             traj_preds,
             traj_targets,
             gt_fut_masks,
             neg_inds):
        """"Get Trajectory mode classification target.
        Args:
            traj_preds (Tensor): MultiModal traj preds with shape (num_box_preds, fut_mode, fut_ts, 2).
            traj_targets (Tensor): Ground truth traj for each pred box with shape (num_box_preds, fut_ts, 2).
            gt_fut_masks (Tensor): Ground truth traj mask with shape (num_box_preds, fut_ts).
            neg_inds (Tensor): Negtive indices with shape (num_box_preds,)

        Returns:
            traj_labels (Tensor): traj cls labels (num_box_preds,).
        """

        cum_traj_preds = traj_preds.cumsum(dim=-2)
        cum_traj_targets = traj_targets.cumsum(dim=-2)

        # Get min pred mode indices.
        # (num_box_preds, fut_mode, fut_ts)
        dist = torch.linalg.norm(cum_traj_targets[:, None, :, :] - cum_traj_preds, dim=-1)
        dist = dist * gt_fut_masks[:, None, :]
        dist = dist[..., -1]
        dist[torch.isnan(dist)] = dist[torch.isnan(dist)] * 0
        traj_labels = torch.argmin(dist, dim=-1)
        traj_labels[neg_inds] = self.fut_mode

        return traj_labels