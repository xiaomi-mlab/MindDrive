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

from sqlite3 import Timestamp
from black import out
import torch
import torch.nn.functional as F
from zmq import PROTOCOL_ERROR_ZMTP_MALFORMED_COMMAND_MESSAGE
from mmcv.utils import auto_fp16
from mmcv.models import DETECTORS
import copy
import os
from mmcv.models.builder import build_head

from mmcv.core import bbox3d2result

from mmcv.models.detectors.mvx_two_stage import MVXTwoStageDetector
from mmcv.models.utils.grid_mask import GridMask

from mmcv.utils.misc import locations

from ...datasets.data_utils.constants import IGNORE_INDEX, WAYPOINT_TOKEN, \
                                                LEFT_WAYPOINT_TOKEN, RIGHT_WAYPOINT_TOKEN, \
                                                STRAIGHT_WAYPOINT_TOKEN, FOLLOW_WAYPOINT_TOKEN, \
                                                CHANGE_LEFT_WAYPOINT_TOKEN, CHANGE_RIGHT_WAYPOINT_TOKEN, \
                                                EGO_WAYPOINT_TOKEN,EGO_PATH_WAYPOINT_TOKEN,MAINTAIN_MODERATE_SPEED,\
                                                STOP,MAINTAIN_SLOW_SPEED,SPEED_UP,SLOW_DOWN,MAIN_FAST_SPEED,SLOW_DOWN_RAPIDLY,\
                                                LANEFOLLOW,STRAIGHT,TURN_LEFT,CHANGE_LANE_LEFT,TURN_RIGHT,CHANGE_LANE_RIGHT
from mmcv.models import builder

from ...utils.llava_llama import LlavaLlamaForCausalLM, add_special_token
from transformers import AutoTokenizer, GenerationConfig

from mmcv.utils.misc import load_model

from ..utils.positional_encoding import pos2posemb2d
import torch.nn as nn
import os
import json
import mmcv
from mmcv.utils.misc import MLN
from mmcv.models.utils.transformer import inverse_sigmoid
from pathlib import Path
import time
import re
import numpy as np
import random
from mmcv.models.dense_heads.planning_head_plugin.metric_stp3 import PlanningMetric
from scipy.optimize import linear_sum_assignment
import cv2
from mmcv.models.utils.vis_utils import format_bbox, show_multicam_bboxes, draw_ld_vis
from mmcv.utils import force_fp32, auto_fp16
from ..utils.freeze_module import freeze_module
from mmcv.models.utils import  DistributionModule, PredictModel, DistributionDecoder1DV2, PredictModelHidden, Bottleneck, SpatialGRU, FuturePrediction,  \
                                CustomTransformerDecoder, CustomTransformerDecoderLayer, SinusoidalPosEmb, gen_sineembed_for_position, \
                                    linear_relu_ln, py_sigmoid_focal_loss
from mmcv.models.bricks import Linear
from mmcv.models.builder import HEADS, build_loss 
import pickle
from diffusers.schedulers import DDIMScheduler
import matplotlib.pyplot as plt
from mmcv.utils.misc import memory_refresh
from mmcv.models.utils import build_transformer
from mmcv.datasets.data_utils.data_utils import  preprocess_qwen2
from io import BytesIO
from stable_baselines3.common.distributions import CategoricalDistribution
import time

SPEED_MAPPING = {
    0: '<maintain_moderate_speed>',
    1: '<stop>',
    2: '<maintain_slow_speed>',
    3: '<speed_up>',
    4: '<slow_down>',
    5: '<maintain_fast_speed>',
    6: '<slow_down_rapidly>'
}

PATH_MAPPING = {
    0: '<turn_left>',    
    1: '<turn_right>',     
    2: '<straight>',       
    3: '<lanefollow>',     
    4: '<change_lane_left>',
    5: '<change_lane_right>'
}

@DETECTORS.register_module()
class Minddrive(MVXTwoStageDetector):
    """Minddrive."""
    def __init__(self,
                 save_path='./results_vlm/',
                 use_grid_mask=False,
                 embed_dims=256,
                 LID=True,
                 position_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                 depth_num=64,
                 depth_start = 1,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 map_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 lm_head=None,
                 tokenizer=None,
                 lm_model_type = None,
                 train_cfg=None,
                 test_cfg=None,
                 stride=16,
                 position_level=0,
                 aux_2d_only=True,
                 frozen=True,
                 use_lora=False,
                 output_text=False,
                 full_train = False,
                 pretrained=None,
                 fp16_infer=False,
                 fp16_eval=False,
                 fp32_infer=False,
                 planning_memory=False,
                 teacher_forcing=False,
                 fut_ts=6,
                 fut_ps =20,
                 freeze_backbone=False,
                 use_col_loss = False,
                 use_gen_token=False,
                 use_meta_action = False,
                 is_decoupling = False,
                 with_bound_loss=True,
                 loss_plan_reg=dict(type='L1Loss', loss_weight=0.25),
                 loss_plan_bound=dict(type='PlanMapBoundLoss', loss_weight=0.1),
                 loss_plan_col=dict(type='PlanCollisionLoss', loss_weight=1.0),
                 loss_vae_gen=dict(type='ProbabilisticLoss', loss_weight=1.0),
                 plan_cls_loss_smooth = False,
                 use_critical_qa=False,
                 plan_anchor_path=None,
                 diff_loss_weight=2.0,
                 ego_fut_mode=20,
                 noise_x_offset=12,
                 noise_x_scale=24,
                 noise_y_offset=10,
                 noise_y_scale=40,
                 qa_pretrain=False,
                 temporal_prompt_input=False,
                 rl_training = False,
                 mix_qa_training=False,
                 open_loop_infer=True):
        super(Minddrive, self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.save_path = save_path
        self.mix_qa_training = mix_qa_training
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.stride = stride
        self.use_col_loss = use_col_loss
        self.position_level = position_level
        self.aux_2d_only = aux_2d_only
        self.lm_model_type = lm_model_type
        self.use_meta_action = use_meta_action
        self.query_pos = nn.Sequential(
            nn.Linear(396, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
        )
        self.fut_ts = fut_ts
        self.fut_ps = fut_ps
        self.time_embedding = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.LayerNorm(embed_dims)
        )

        self.ego_pose_pe = MLN(156)
        self.rl_training = rl_training
        self.pts_bbox_head.query_pos = self.query_pos
        self.pts_bbox_head.time_embedding = self.time_embedding
        self.pts_bbox_head.ego_pose_pe = self.ego_pose_pe
        self.open_loop_infer = open_loop_infer
        if map_head is not None:
            self.map_head = builder.build_head(map_head)
            self.map_head.query_pos = self.query_pos
            self.map_head.time_embedding = self.time_embedding
            self.map_head.ego_pose_pe = self.ego_pose_pe

        if tokenizer is not None:
            self.tokenizer =  AutoTokenizer.from_pretrained(tokenizer,
                                        model_max_length=2048,
                                        padding_side="right",
                                        use_fast=False,
                                        )
            if self.lm_model_type == 'llama_v3':
                self.tokenizer.eos_token_id = self.tokenizer.convert_tokens_to_ids('<|eot_id|>')
                self.tokenizer.pad_token = self.tokenizer.eos_token
            elif self.lm_model_type == 'qwen2' or self.lm_model_type == 'qwen25_3B':
                self.tokenizer.eos_token_id = self.tokenizer.convert_tokens_to_ids('<|im_end|>')
            else:
                self.tokenizer.pad_token = self.tokenizer.unk_token
        else:
            self.tokenizer = None
        
        self.position_range = nn.Parameter(torch.tensor(
            position_range), requires_grad=False)
        
        if LID:
            index  = torch.arange(start=0, end=depth_num, step=1).float()
            index_1 = index + 1
            bin_size = (self.position_range[3] - depth_start) / (depth_num * (1 + depth_num))
            coords_d = depth_start + bin_size * index * index_1
        else:
            index  = torch.arange(start=0, end=depth_num, step=1).float()
            bin_size = (self.position_range[3] - depth_start) / depth_num
            coords_d = depth_start + bin_size * index

        self.coords_d = nn.Parameter(coords_d, requires_grad=False)

        self.position_encoder = nn.Sequential(
                nn.Linear(depth_num*3, embed_dims*4),
                nn.ReLU(),
                nn.Linear(embed_dims*4, embed_dims),
            )
        
        use_critical_qa = use_critical_qa or qa_pretrain
        self.qa_pretrain = qa_pretrain
        if lm_head is not None:
            lm_kwargs = dict(use_gen_token=use_gen_token,use_critical_qa=use_critical_qa, use_meta_action=self.use_meta_action)
            self.lm_head = load_model(lm_head, use_lora, frozen, lm_kwargs, fp16_infer, full_train, self.lm_model_type, adapter_names=["action_expert", "decision_expert"])          

        self.is_decoupling = is_decoupling
        if use_meta_action:
            add_special_token([MAINTAIN_MODERATE_SPEED,STOP,MAINTAIN_SLOW_SPEED,SPEED_UP,SLOW_DOWN,MAIN_FAST_SPEED,\
                SLOW_DOWN_RAPIDLY,LANEFOLLOW,STRAIGHT,TURN_LEFT,CHANGE_LANE_LEFT,TURN_RIGHT,CHANGE_LANE_RIGHT], tokenizer = self.tokenizer, model = self.lm_head) 
            meta_action_token_idx = self.tokenizer([MAINTAIN_MODERATE_SPEED,STOP,MAINTAIN_SLOW_SPEED,SPEED_UP,SLOW_DOWN,MAIN_FAST_SPEED,\
                SLOW_DOWN_RAPIDLY,LANEFOLLOW,STRAIGHT,TURN_LEFT,CHANGE_LANE_LEFT,TURN_RIGHT,CHANGE_LANE_RIGHT], add_special_tokens=False).input_ids     
            self.meta_action_token_idx = [item[0] for item in meta_action_token_idx]
            self.lm_head.config.meta_action_token_idx = self.meta_action_token_idx
        
            self.SPT_MAPPING = {
                        '<maintain_moderate_speed>': 0,
                        '<stop>':1,
                        '<maintain_slow_speed>':2,
                        '<speed_up>':3,
                        '<slow_down>':4,
                        '<maintain_fast_speed>':5,
                        '<slow_down_rapidly>':6}
            self.PA_MAPPING= {
                    '<lanefollow>': 0,
                    '<straight>':1,
                    '<turn_left>': 2,
                    '<change_lane_left>':3,
                    '<turn_right>': 4,
                    '<change_lane_right>':5, 
                    }
            self.action_distribution = CategoricalDistribution(action_dim=7)
        if use_gen_token:
            add_special_token([EGO_WAYPOINT_TOKEN], tokenizer = self.tokenizer, model = self.lm_head)
            self.lm_head.config.waypoint_token_idx = self.tokenizer(EGO_WAYPOINT_TOKEN, add_special_tokens=False).input_ids[0]
            if self.is_decoupling:
                waypoint_token_idx = []
                waypoint_token_idx.append(self.tokenizer(EGO_WAYPOINT_TOKEN, add_special_tokens=False).input_ids[0]) # 原来的waypoint
                add_special_token([EGO_PATH_WAYPOINT_TOKEN], tokenizer = self.tokenizer, model = self.lm_head)
                waypoint_token_idx.append(self.tokenizer(EGO_PATH_WAYPOINT_TOKEN, add_special_tokens=False).input_ids[0])
                self.lm_head.config.waypoint_token_idx = waypoint_token_idx

            self.SPEED_MAPPING = {
                    'maintain moderate speed': 0,
                    'stop':1,
                    'maintain slow speed':2,
                    'speed up':3,
                    'slow down':4,
                    'maintain fast speed':5,
                    'slow down rapidly':6}
            self.PATH_MAPPING= {
                'lanefollow': 0,
                'straight':1,
                'turn left': 2,
                'change lane left':3,
                'turn right': 4,
                'change lane right':5, 
                }
        self.use_gen_token = use_gen_token
        

        if self.rl_training:
            value_lm_kwargs = dict(use_gen_token=use_gen_token,use_critical_qa=use_critical_qa, use_meta_action=self.use_meta_action, value_net = True)
            self.value_net = load_model(lm_head, use_lora, frozen, value_lm_kwargs, fp16_infer, full_train, self.lm_model_type)

            self.value_net.resize_token_embeddings(len(self.tokenizer))
            if self.lm_model_type=='qwen25_3B':
                self.value_net_pro = torch.nn.Linear(2048, 1)
            else:
                self.value_net_pro = torch.nn.Linear(896, 1)  # qwen 的隐藏层为896
            torch.nn.init.xavier_uniform_(self.value_net_pro.weight)        
            self.value_net.config.waypoint_token_idx = waypoint_token_idx
            self.value_net.config.meta_action_token_idx = self.meta_action_token_idx
            
        if self.use_gen_token:
            
            self.layer_dim = 4
            self.with_bound_loss = with_bound_loss
            self.with_cur = True
            if self.lm_model_type == 'llama_v3':
                self.present_distribution_in_channels = 2048
            elif self.lm_model_type == 'qwen2':
                self.present_distribution_in_channels = 896
            elif self.lm_model_type == 'qwen25_3B':
                self.present_distribution_in_channels = 2048
            else:
                self.present_distribution_in_channels = 4096
            self.future_distribution_in_channels = self.present_distribution_in_channels+12
            self.now_pred_in_channels = 64
            self.PROBABILISTIC = True
            self.latent_dim = 32
            self.MIN_LOG_SIGMA = -5.0
            self.MAX_LOG_SIGMA = 5.0
            self.FUTURE_DIM = 6
            self.N_GRU_BLOCKS = 3
            self.N_RES_LAYERS = 3
            self.embed_dims = embed_dims
            if self.use_meta_action:
                self.ego_fut_mode = 7
                self.pw_ego_fut_mode = 6
            else:
                self.ego_fut_mode = 6

            self.present_distribution = DistributionModule(
                self.present_distribution_in_channels,
                self.latent_dim,
                min_log_sigma=self.MIN_LOG_SIGMA,
                max_log_sigma=self.MAX_LOG_SIGMA,
            )

            self.future_distribution = DistributionModule(
                self.future_distribution_in_channels,
                self.latent_dim,
                min_log_sigma=self.MIN_LOG_SIGMA,
                max_log_sigma=self.MAX_LOG_SIGMA,
            )

            assert self.present_distribution_in_channels%self.layer_dim == 0
            self.predict_model = PredictModel(
                in_channels=self.latent_dim,
                out_channels=self.present_distribution_in_channels,
                hidden_channels=int(self.present_distribution_in_channels/self.layer_dim),
                num_layers=self.layer_dim
            )
            ego_fut_decoder = []
            for _ in range(2):
                ego_fut_decoder.append(Linear(self.present_distribution_in_channels*2, self.present_distribution_in_channels*2))
                ego_fut_decoder.append(nn.ReLU())
            ego_fut_decoder.append(Linear(self.present_distribution_in_channels*2, self.ego_fut_mode*2))
            self.ego_fut_decoder = nn.Sequential(*ego_fut_decoder)

            self.loss_plan_reg = build_loss(loss_plan_reg)
            self.loss_plan_bound = build_loss(loss_plan_bound)
            if self.use_col_loss:
                self.loss_plan_col = build_loss(loss_plan_col)
            self.loss_vae_gen = build_loss(loss_vae_gen)
            if self.is_decoupling:
                self.pw_present_distribution = DistributionModule(
                    self.present_distribution_in_channels,
                    self.latent_dim,
                    min_log_sigma=self.MIN_LOG_SIGMA,
                    max_log_sigma=self.MAX_LOG_SIGMA,
                    )
                self.pw_future_distribution_in_channels = self.present_distribution_in_channels+ 40
                self.pw_future_distribution = DistributionModule(
                    self.pw_future_distribution_in_channels,
                    self.latent_dim,
                    min_log_sigma=self.MIN_LOG_SIGMA,
                    max_log_sigma=self.MAX_LOG_SIGMA,
                )
                self.pw_predict_model = PredictModel(
                in_channels=self.latent_dim,
                out_channels=self.present_distribution_in_channels,
                hidden_channels=int(self.present_distribution_in_channels/self.layer_dim),
                num_layers=self.layer_dim
                    )
                pw_ego_fut_decoder = []
                
                for _ in range(2):
                    pw_ego_fut_decoder.append(Linear(self.present_distribution_in_channels*2, self.present_distribution_in_channels*2))
                    pw_ego_fut_decoder.append(nn.ReLU())
                pw_ego_fut_decoder.append(Linear(self.present_distribution_in_channels*2, self.pw_ego_fut_mode*2))
                self.pw_ego_fut_decoder = nn.Sequential(*pw_ego_fut_decoder)
        self.test_flag = False
        self.planning_metric = None
        self.output_text = output_text
        if fp16_infer:
            self.img_backbone.half()
        self.fp16_infer = fp16_infer
        self.fp16_eval = fp16_eval
        assert fp16_infer if fp16_eval else True
        self.fp32_infer = fp32_infer
        assert not fp16_infer if fp32_infer else True

        if planning_memory:
            self.pmemroy = True
            self.planning_memory = None
            self.teacher_forcing = teacher_forcing
        else:
            self.pmemroy = False
            self.teacher_forcing = False


        self.freeze_backbone = freeze_backbone
        self.temporal_prompt_input = temporal_prompt_input

    def train(self, mode, *args, **kwargs):
        super().train(mode, *args, **kwargs)
        
        if mode and self.freeze_backbone:
            for shared_module_name in self.train_cfg.get('backbone_weights'):
                items = shared_module_name.split('.')
                shared_module = self
                for item in items:
                    shared_module = getattr(shared_module, item)
                freeze_module(shared_module)
                print(f'Freeze: {shared_module_name}')
        return self

    @property
    def with_map_head(self):
        """bool: Whether the detector has a map head."""
        return hasattr(self,
                       'map_head') and self.map_head is not None
        
    
    @property
    def with_lm_head(self):
        """bool: Whether the detector has a lm head."""
        return hasattr(self,
                       'lm_head') and self.lm_head is not None
        
    def extract_img_feat(self, img):
        """Extract features of images."""
        B = img.size(0)

        if img is not None:
            if img.dim() == 6:
                img = img.flatten(1, 2)
            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        BN, C, H, W = img_feats[self.position_level].size()

        img_feats_reshaped = img_feats[self.position_level].view(B, int(BN/B), C, H, W)


        return img_feats_reshaped


    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_feat(self, img):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img)
        return img_feats


    def prepare_location(self, img_metas, **data):
        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        bs, n = data['img_feats'].shape[:2]
        x = data['img_feats'].flatten(0, 1)
        location = locations(x, self.stride, pad_h, pad_w)[None].repeat(bs*n, 1, 1, 1)
        return location

    def forward_roi_head(self, location, **data):
        if (self.aux_2d_only and not self.training) or not self.with_img_roi_head:
            return {'topk_indexes':None}
        else:
            outs_roi = self.img_roi_head(location, **data)
            return outs_roi


    def position_embeding(self, data, memory_centers, img_metas):
        eps = 1e-5
        BN, H, W, _ = memory_centers.shape
        B = data['cam_intrinsic'].size(0)

        intrinsic = torch.stack([data['cam_intrinsic'][..., 0, 0], data['cam_intrinsic'][..., 1, 1]], dim=-1)
        intrinsic = torch.abs(intrinsic) / 1e3
        intrinsic = intrinsic.repeat(1, H*W, 1).view(B, -1, 2)
        LEN = intrinsic.size(1)

        num_sample_tokens = LEN

        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        memory_centers[..., 0] = memory_centers[..., 0] * pad_w
        memory_centers[..., 1] = memory_centers[..., 1] * pad_h

        D = self.coords_d.shape[0]

        memory_centers = memory_centers.detach().view(B, LEN, 1, 2)
        topk_centers = memory_centers.repeat(1, 1, D, 1)
        coords_d = self.coords_d.view(1, 1, D, 1).repeat(B, num_sample_tokens, 1 , 1)
        coords = torch.cat([topk_centers, coords_d], dim=-1)
        coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)
        coords[..., :2] = coords[..., :2] * torch.maximum(coords[..., 2:3], torch.ones_like(coords[..., 2:3])*eps)

        coords = coords.unsqueeze(-1)

        img2lidars = data['lidar2img'].inverse()
        img2lidars = img2lidars.view(BN, 1, 1, 4, 4).repeat(1, H*W, D, 1, 1).view(B, LEN, D, 4, 4)

        coords3d = torch.matmul(img2lidars, coords).squeeze(-1)[..., :3]
        coords3d[..., 0:3] = (coords3d[..., 0:3] - self.position_range[0:3]) / (self.position_range[3:6] - self.position_range[0:3])
        coords3d = coords3d.reshape(B, -1, D*3)
      
        pos_embed  = inverse_sigmoid(coords3d)
        coords_position_embeding = self.position_encoder(pos_embed)

        return coords_position_embeding

    def forward_pts_train(self,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          gt_attr_labels,
                          map_gt_bboxes_3d,
                          map_gt_labels_3d,   
                          img_metas,
                          input_ids, 
                          vlm_labels, 
                          vlm_attn_mask,
                          ego_fut_trajs,
                          **data):
        """Forward function for point cloud branch.
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
        Returns:
            dict: Losses of each branch.
        """
        B = data['img'].shape[0]
        location = self.prepare_location(img_metas, **data) # (6, 40, 40, 2)

        pos_embed = self.position_embeding(data, location, img_metas) # (1, 9600, 256)
        losses = dict()

        if self.with_map_head:
            outs_lane, map_query = self.map_head(img_metas, pos_embed, **data)
            vision_embeded_map = map_query.clone()
            device = gt_labels_3d[0].device
            map_gt_vecs_list = copy.deepcopy(map_gt_bboxes_3d)
            lane_pts = [F.pad(map_gt_bboxes.fixed_num_sampled_points.to(device),(0,1)) for map_gt_bboxes in map_gt_vecs_list]
            loss_inputs = [lane_pts, map_gt_labels_3d, outs_lane, img_metas]

            if False:
                import pickle
                with open('lane_pts.pkl', 'wb') as file:
                    pickle.dump(lane_pts, file)
            losses.update(self.map_head.loss(*loss_inputs))

        if self.with_pts_bbox:
            outs_bbox, det_query = self.pts_bbox_head(img_metas, pos_embed, outs_lane, **data)
            vision_embeded_obj = det_query.clone()
            loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs_bbox, gt_attr_labels]
            if self.pts_bbox_head.pred_traffic_light_state:
                loss_inputs.append(data['traffic_state'])
                loss_inputs.append(data['traffic_state_mask'])
            if self.use_col_loss:
                loss, agent_outs = self.pts_bbox_head.loss(*loss_inputs)
            else:
                loss = self.pts_bbox_head.loss(*loss_inputs)
            losses.update(loss)
        

        if self.with_lm_head:
            if self.use_gen_token:
                vision_embeded = torch.cat([vision_embeded_obj, vision_embeded_map], dim=1) 
                vlm_loss, ego_feature = self.lm_head(input_ids=input_ids, attention_mask=vlm_attn_mask, labels=vlm_labels, images=vision_embeded, use_cache=False, return_ego_feature=True)
                if self.mix_qa_training:
                    dummy_ego_feature = self.lm_head.get_model().embed_tokens(torch.tensor([[self.lm_head.config.waypoint_token_idx] for _ in range(B)]).cuda())
                    dummy_ego_feature = dummy_ego_feature.squeeze(1)
                    waypoint_token_indices = self.lm_head.config.waypoint_token_idx
                    waypoint_tokens = torch.tensor(waypoint_token_indices, device=input_ids.device)
                    mask_per_token = [
                        (input_ids == token).any(dim=-1, keepdim=True) 
                        for token in waypoint_tokens
                    ]
                    valid_input_mask = torch.all(torch.cat(mask_per_token, dim=-1), dim=-1) 
                    updated_ego_feature = dummy_ego_feature.clone()
                    if self.lm_model_type == 'qwen2':
                        updated_ego_feature[valid_input_mask] = ego_feature.reshape(-1, len(self.lm_head.config.waypoint_token_idx), 896)  # 
                    elif self.lm_model_type == 'qwen25_3B':  
                        updated_ego_feature[valid_input_mask] = ego_feature.reshape(-1, len(self.lm_head.config.waypoint_token_idx), 2048)  
                    ego_feature = updated_ego_feature
                    data['ego_fut_masks'][:,0,0] *= valid_input_mask.unsqueeze(-1)
                losses.update(vlm_loss=vlm_loss[0])
                if self.is_decoupling:
                    if self.lm_model_type == 'qwen2':
                        ego_feature = ego_feature.reshape(B, -1, 896) 
                    elif self.lm_model_type == 'qwen25_3B':
                        ego_feature = ego_feature.reshape(B, -1, 2048)
                    current_states = ego_feature[:,0].unsqueeze(1)
                    pw_current_states = ego_feature[:,1].unsqueeze(1)
                else:
                    current_states = ego_feature.unsqueeze(1)

                distribution_comp = {}

                noise = None
                self.fut_ts = 6
                self.fut_ps = 20
                if self.training:
                    future_distribution_inputs = ego_fut_trajs.reshape(B, ego_fut_trajs.shape[1], -1)
                    if self.is_decoupling:
                        pw_future_distribution_inputs = data['path_points_future'].reshape(B, data['path_points_future'].shape[1], -1)

                if self.PROBABILISTIC:
                    sample, output_distribution = self.distribution_forward(
                        current_states, future_distribution_inputs, noise
                    )
                    distribution_comp = {**distribution_comp, **output_distribution}
                    if self.is_decoupling:
                        pw_distribution_comp = {}
                        pw_sample, pw_output_distribution = self.pw_distribution_forward(
                            pw_current_states, pw_future_distribution_inputs, noise
                        )
                        pw_distribution_comp = {**pw_distribution_comp, **pw_output_distribution}

                if self.is_decoupling:
                    hidden_states = ego_feature[:,0].unsqueeze(1)
                    pw_hidden_states = ego_feature[:,1].unsqueeze(1)
                    states_hs, future_states_hs = \
                        self.future_states_predict(B, sample, hidden_states, current_states)
                    pw_states_hs, pw_future_states_hs = \
                        self.pw_future_states_predict(B, pw_sample, pw_hidden_states, pw_current_states)
                    ego_query_hs = \
                        states_hs[:, :, 0, :].unsqueeze(1).permute(0, 2, 1, 3)
                    pw_ego_query_hs = \
                        pw_states_hs[:, :, 0, :].unsqueeze(1).permute(0, 2, 1, 3)     
                    ego_fut_trajs_list = []
                    for i in range(self.fut_ts):
                        outputs_ego_trajs = self.ego_fut_decoder(ego_query_hs[i]).reshape(B, self.ego_fut_mode, 2)
                        ego_fut_trajs_list.append(outputs_ego_trajs) 

                    pw_ego_fut_trajs_list = []
                    for i in range(self.fut_ps):
                        pw_outputs_ego_trajs = self.pw_ego_fut_decoder(pw_ego_query_hs[i]).reshape(B, self.pw_ego_fut_mode, 2)
                        pw_ego_fut_trajs_list.append(pw_outputs_ego_trajs)  
                    pw_ego_fut_preds = torch.stack(pw_ego_fut_trajs_list, dim=2)                    
                else:
                    hidden_states = ego_feature.unsqueeze(1)
                    states_hs, future_states_hs = \
                        self.future_states_predict(B, sample, hidden_states, current_states)
                    ego_query_hs = \
                        states_hs[:, :, 0, :].unsqueeze(1).permute(0, 2, 1, 3)
                
                    ego_fut_trajs_list = []
                    for i in range(self.fut_ts):
                        outputs_ego_trajs = self.ego_fut_decoder(ego_query_hs[i]).reshape(B, self.ego_fut_mode, 2)
                        ego_fut_trajs_list.append(outputs_ego_trajs)

                ego_fut_preds = torch.stack(ego_fut_trajs_list, dim=2)

                lane_scores = outs_lane['all_lane_cls_one2one'][-1]
                lane_preds = outs_lane['all_lane_preds_one2one'][-1]
                for p in range(self.map_head.n_control):
                    lane_preds[..., 3 * p].clamp_(min=self.map_head.pc_range[0], max=self.map_head.pc_range[3])
                    lane_preds[..., 3 * p + 1].clamp_(min=self.map_head.pc_range[1], max=self.map_head.pc_range[4])
                lane_preds = lane_preds.reshape(lane_preds.shape[0],lane_preds.shape[1],-1,3)[...,:2]
                
                if self.with_bound_loss:
                    if self.use_meta_action:
                        loss_plan_input = [ego_fut_preds, ego_fut_trajs[:,0], data['ego_fut_masks'][:,0,0], data['cmd_speed'][:,0,0], lane_preds, lane_scores]
                    else:
                        loss_plan_input = [ego_fut_preds, ego_fut_trajs[:,0], data['ego_fut_masks'][:,0,0], data['ego_fut_cmd'][:,0,0], lane_preds, lane_scores]
                else:
                    loss_plan_input = [ego_fut_preds, ego_fut_trajs[:,0], data['ego_fut_masks'][:,0,0], data['ego_fut_cmd'][:,0,0]]
                
                if self.use_col_loss:
                    loss_planning_dict = self.loss_planning(*loss_plan_input, **agent_outs)
                else:
                    loss_planning_dict = self.loss_planning(*loss_plan_input)
                losses.update(loss_planning_dict)
                loss_vae_gen = self.loss_vae_gen(distribution_comp, data['ego_fut_masks'][:,0,0])
                loss_vae_gen = torch.nan_to_num(loss_vae_gen)
                losses.update(loss_vae_gen=loss_vae_gen)
                if self.is_decoupling:
                    if self.with_bound_loss:
                        if self.use_meta_action:
                            loss_plan_input = [pw_ego_fut_preds, data['path_points_future'][:,0], data['path_future_mask'][:,0,0], data['cmd_path'][:,0,0], lane_preds, lane_scores]
                        else:
                            loss_plan_input = [pw_ego_fut_preds, data['path_points_future'][:,0], data['path_future_mask'][:,0,0], data['ego_fut_cmd'][:,0,0], lane_preds, lane_scores]
                    else:
                        loss_plan_input = [pw_ego_fut_preds, data['path_points_future'][:,0], data['path_future_mask'][:,0,0], data['ego_fut_cmd'][:,0,0]]

                    loss_planning_dict = self.loss_planning(*loss_plan_input, path_loss=True) 
                    loss_plan_dict = {}
                    loss_plan_dict = {f"pw_{k}": v for k, v in loss_planning_dict.items()}
                    losses.update(loss_plan_dict)

                    loss_vae_gen = self.loss_vae_gen(pw_distribution_comp, data['path_future_mask'][:,0,0])
                    loss_vae_gen = torch.nan_to_num(loss_vae_gen)
                    losses.update(loss_pw_vae_gen=loss_vae_gen)          
            else:
                waypoint = None
                vision_embeded = torch.cat([vision_embeded_obj, vision_embeded_map], dim=1) 
                vlm_loss= self.lm_head(input_ids=input_ids, attention_mask=vlm_attn_mask, labels=vlm_labels, images=vision_embeded, use_cache=False)
                losses.update(vlm_loss=vlm_loss[0])
            
        if os.getenv('DEBUG_SHOW_PRED', None) == "1":
            show_dir = os.getenv('DEBUG_SHOW_PRED_DIR', None)
            _ = self.show_results(data['img'], img_metas, data, ego_fut_trajs=ego_fut_trajs,
                waypoint=waypoint, outs_bbox=outs_bbox, 
                outs_lane=outs_lane, use_gt=False, show_dir=show_dir)

        return losses
        
    def forward(self, data, return_loss=True, is_rl_training=False):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            # return self.forward_train(**data)
            losses = self.forward_train(**data)
            loss, log_vars = self._parse_losses(losses)
            outputs = dict(
                loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))
            return outputs
        elif is_rl_training:
            policy_action_log_prob, action_language_log_probs, values= self.forward_ppo_train(**data)
            return policy_action_log_prob, action_language_log_probs, values
        else:
            return self.forward_test(**data)

    def forward_ppo_train(self, inputs_embeds=None,
                                new_input_ids=None):
                                
        inputs_embeds = inputs_embeds.float().clone().detach().requires_grad_(True)
        meta_action_info = {
            'inputs_embeds':inputs_embeds,
            'new_input_ids' : new_input_ids,
        } 
        self.lm_head.set_adapter("decision_expert") # only_train decision_expert lora
        policy_action_log_prob, action_language_log_probs = self.lm_head.forward_rl(meta_action_info) # (1, 9)
        hidden_states = self.value_net.forward_rl_value(meta_action_info)
        values = self.value_net_pro(hidden_states)
        
        return policy_action_log_prob, action_language_log_probs, values


    def forward_train(self,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_attr_labels= None,
                      map_gt_bboxes_3d=None,
                      map_gt_labels_3d=None,
                      input_ids=None,
                      vlm_labels=None,
                      vlm_attn_mask = None,
                      gt_bboxes_ignore=None,
                      map_gt_bboxes_ignore=None,
                      ego_fut_trajs = None,
                      **data):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        if self.test_flag: #for interval evaluation
            self.pts_bbox_head.reset_memory()
            self.test_flag = False
        if self.tokenizer is not None:
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids, # [(76,)]
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id) # (1, 76)
            
            vlm_labels = torch.nn.utils.rnn.pad_sequence(vlm_labels, # [(76,)]
                                                    batch_first=True,
                                                    padding_value=IGNORE_INDEX) # (1, 76)
            vlm_attn_mask = torch.nn.utils.rnn.pad_sequence(vlm_attn_mask, # [(76,)]
                                                    batch_first=True,
                                                    padding_value=0) # (1, 76)
            
            input_ids = input_ids[:, :self.tokenizer.model_max_length] # 2048
            vlm_labels = vlm_labels[:, :self.tokenizer.model_max_length] # 2048
            vlm_attn_mask = vlm_attn_mask[:, :self.tokenizer.model_max_length].bool()
        else:
            input_ids = None
            vlm_labels = None
            vlm_attn_mask = None
        img_metas = [img_meta[0] for img_meta in img_metas]

        data['img_feats'] = self.extract_feat(data['img'])
        losses = self.forward_pts_train(gt_bboxes_3d, gt_labels_3d, gt_attr_labels,map_gt_bboxes_3d, map_gt_labels_3d, img_metas,input_ids, vlm_labels, vlm_attn_mask, ego_fut_trajs,**data)

        return losses
  
  
    def forward_test(self, img_metas, **data):
        if not self.test_flag: #for interval evaluation
            if self.with_pts_bbox:
                self.pts_bbox_head.reset_memory()
            if self.with_map_head:
                self.map_head.reset_memory()
            self.test_flag = True
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        for key in data:
            if key not in ['img', 'input_ids','gt_bboxes_3d','vlm_labels']:
                data[key] = data[key][0][0].unsqueeze(0)
            else:
                data[key] = data[key][0]
        return self.simple_test(img_metas[0], **data)

    def simple_test_pts(self, img_metas, **data):
        """Test function of point cloud branch."""
        B = 1
        mapped_class_names = [
            'car', 'truck', 'construction_vehicle', 'bus',
            'trailer', 'barrier', 'motorcycle', 'bicycle', 
            'pedestrian', 'traffic_cone'
        ]
        #check
        location = self.prepare_location(img_metas, **data)
        outs_roi = self.forward_roi_head(location, **data)
        pos_embed = self.position_embeding(data, location, img_metas)
        bbox_results = []
        lane_results = None
        if self.with_map_head:
            outs, map_query = self.map_head(img_metas, pos_embed, **data)
            vision_embeded_map = map_query.clone()
            lane_results = self.map_head.get_bboxes(outs, img_metas)

        if self.with_pts_bbox:
            outs, det_query = self.pts_bbox_head(img_metas, pos_embed, outs, **data)
            vision_embeded_obj = det_query.clone()
            if self.use_col_loss:
                bbox_list = self.pts_bbox_head.get_motion_bboxes(
                outs, img_metas)
                for bboxes, scores, labels, trajs in bbox_list:
                    bbox_result = bbox3d2result(bboxes, scores, labels)
                    bbox_result['trajs_3d'] = trajs.cpu()
                    bbox_results.append(bbox_result)
            else:
                bbox_list = self.pts_bbox_head.get_bboxes(
                    outs, img_metas)
                for bboxes, scores, labels in bbox_list:
                    bbox_results.append(bbox3d2result(bboxes, scores, labels))
    
        generated_text = []
        metric_dict = {}

        if not (self.fp16_infer or self.fp32_infer) or self.fp16_eval :
            gt_attr_label = data['gt_attr_labels'][0].to('cpu')
            gt_bbox = data['gt_bboxes_3d'][0]
            fut_valid_flag = bool(data['fut_valid_flag'][0])
            gt_label = data['gt_labels_3d'][0].to('cpu')
            if self.use_col_loss:
                score_threshold = 0.6
                with torch.no_grad():
                    c_bbox_results = copy.deepcopy(bbox_results)
                    bbox_result = c_bbox_results[0]
                    mask = bbox_result['scores_3d'] > score_threshold
                    bbox_result['boxes_3d'] = bbox_result['boxes_3d'][mask]
                    bbox_result['scores_3d'] = bbox_result['scores_3d'][mask]
                    bbox_result['labels_3d'] = bbox_result['labels_3d'][mask]
                    bbox_result['trajs_3d'] = bbox_result['trajs_3d'][mask]

                    matched_bbox_result = self.assign_pred_to_gt_vip3d(
                        bbox_result, gt_bbox, gt_label)

                    metric_dict = self.compute_motion_metric_vip3d(
                            gt_bbox, gt_label, gt_attr_label, bbox_result,
                            matched_bbox_result, mapped_class_names)
            
        if self.with_lm_head:
            history_input_output_id = []
            meta_action = {}
            meta_action_info ={}
            vision_embeded = torch.cat([vision_embeded_obj, vision_embeded_map], dim=1) # (1, 513, 4096)
            for i, input_ids in enumerate(data['input_ids'][0]):
                input_ids = input_ids.unsqueeze(0)
                special_token_inputs = False
                special_meta_action_inputs = False
                if not self.qa_pretrain:
                    if hasattr(self.lm_head.config,'waypoint_token_idx'):
                        if isinstance(self.lm_head.config.waypoint_token_idx,list):
                            for sptoken in self.lm_head.config.waypoint_token_idx:
                                if sptoken in input_ids:
                                    special_token_inputs = True
                                    break
                        else:
                            special_token_inputs = self.lm_head.config.waypoint_token_idx in input_ids
                        if isinstance(self.meta_action_token_idx, list):
                            for sptoken in self.meta_action_token_idx:
                                if sptoken in input_ids:
                                    special_meta_action_inputs = True
                                    break
                if self.use_gen_token and special_meta_action_inputs:
                    self.lm_head.set_adapter("decision_expert")

                    action_logits, inputs_embeds, new_input_ids = self.lm_head.inference_action_distribution(
                        inputs=input_ids,
                        images=vision_embeded,
                        do_sample=True,
                        temperature=0.1,
                        top_p=0.75,
                        num_beams=1,
                        max_new_tokens=320,
                        use_cache=True,
                        )
                    
                    meta_action_info = {
                                'inputs_embeds':inputs_embeds,
                                'new_input_ids' : new_input_ids,
                            }
                    self.action_distribution.proba_distribution(action_logits=action_logits)
                    if self.rl_training:
                        action_speed_idx = self.action_distribution.sample().item()
                    else:
                        action_speed_idx = torch.argmax(action_logits).item()
                    speed_command = SPEED_MAPPING.get(action_speed_idx, '<unknown_speed>')

                    std_cmd_tensors = data['ego_fut_cmd'][:, 0, 0] 
                    
                    path_command = None
                    for cmd_tensor in std_cmd_tensors:
                        path_idx = torch.argmax(cmd_tensor).item()
                        path_command = PATH_MAPPING.get(path_idx, '<unknown_path>')


                    if self.rl_training:
                        hidden_states = self.value_net.forward_rl_value(meta_action_info)
                        values = self.value_net_pro(hidden_states)
                        ppo_info = {
                            'reference_action_log_prob': action_logits.squeeze(-1).cpu(),
                            'values':values.cpu()
                        }

                    if False:
                        planning_qa = [
                            [{"from": 'human',
                            "value": "What actions should the car be taking?"},
                            {"from": 'gpt',
                            "value": f"The car should be {speed_command} {path_command}<|im_end|>"}]
                            ]
                        
                        vqa_anno = [item for pair in planning_qa for item in pair]
                        prompt = "You are driving a car."
                        vqa_anno[0]['value'] = '<image>'+ '\n' + prompt + vqa_anno[0]['value']
                        vqa_converted = preprocess_qwen2([vqa_anno], self.tokenizer, True, training_mode=False, only_one_system_prompt = False)
                        input_ids = vqa_converted['input_ids']
                        # context_input_ids = torch.cat((input_ids[0].to('cuda'), data['input_ids'][0][-1]),dim=-1)
                        history_input_output_id.append(input_ids[0].unsqueeze(0).to(action_logits.device))
                
                    self.lm_head.set_adapter("action_expert")

                elif self.use_gen_token and special_token_inputs: # must be the final round conversation
                    history_input_output_id.append(input_ids)
                    context_input_ids = torch.cat(history_input_output_id,dim=-1)
                    ego_feature = self.lm_head.inference_waypoints(
                        inputs=context_input_ids,
                        images=vision_embeded,
                        do_sample=True,
                        temperature=0.1,
                        top_p=0.75,
                        num_beams=1,
                        max_new_tokens=320,
                        use_cache=True,
                        return_ego_feature=True
                    )
                    ego_feature = ego_feature.to(torch.float32)
                    if self.is_decoupling:
                        if self.lm_model_type == 'qwen2':
                            ego_feature = ego_feature.reshape(B, -1, 896)
                        elif self.lm_model_type == 'qwen25_3B':
                            ego_feature = ego_feature.reshape(B, -1, 2048)
                        current_states = ego_feature[:,0].unsqueeze(1)
                        pw_current_states = ego_feature[:,1].unsqueeze(1)

                    else:
                        current_states = ego_feature.unsqueeze(1)

                    
                    distribution_comp = {}
                    noise = None
                    self.fut_ts = 6
                    if self.PROBABILISTIC:
                        # Do probabilistic computation
                        sample, output_distribution = self.distribution_forward(
                            current_states, None, noise
                        )
                        distribution_comp = {**distribution_comp, **output_distribution}
                        if self.is_decoupling:
                            pw_distribution_comp = {}
                            pw_sample, pw_output_distribution = self.pw_distribution_forward(
                                pw_current_states, None, noise
                            )
                            pw_distribution_comp = {**pw_distribution_comp, **pw_output_distribution}

                    if self.is_decoupling:
                        hidden_states = ego_feature[:,0].unsqueeze(1)
                        pw_hidden_states = ego_feature[:,1].unsqueeze(1)
                        states_hs, future_states_hs = \
                            self.future_states_predict(B, sample, hidden_states, current_states)
                        pw_states_hs, pw_future_states_hs = \
                            self.pw_future_states_predict(B, pw_sample, pw_hidden_states, pw_current_states)
                        ego_query_hs = \
                            states_hs[:, :, 0, :].unsqueeze(1).permute(0, 2, 1, 3)
                        pw_ego_query_hs = \
                            pw_states_hs[:, :, 0, :].unsqueeze(1).permute(0, 2, 1, 3)     
                        ego_fut_trajs_list = []
                        for i in range(self.fut_ts):
                            outputs_ego_trajs = self.ego_fut_decoder(ego_query_hs[i]).reshape(B, self.ego_fut_mode, 2)
                            ego_fut_trajs_list.append(outputs_ego_trajs) 

                        pw_ego_fut_trajs_list = []
                        for i in range(self.fut_ps):
                            pw_outputs_ego_trajs = self.pw_ego_fut_decoder(pw_ego_query_hs[i]).reshape(B, self.pw_ego_fut_mode, 2)
                            pw_ego_fut_trajs_list.append(pw_outputs_ego_trajs)  
                        pw_ego_fut_preds = torch.stack(pw_ego_fut_trajs_list, dim=2)                    
                    else:
                        hidden_states = ego_feature.unsqueeze(1)
                        states_hs, future_states_hs = \
                            self.future_states_predict(B, sample, hidden_states, current_states)
                        ego_query_hs = \
                            states_hs[:, :, 0, :].unsqueeze(1).permute(0, 2, 1, 3)
                    
                        ego_fut_trajs_list = []
                        for i in range(self.fut_ts):
                            outputs_ego_trajs = self.ego_fut_decoder(ego_query_hs[i]).reshape(B, self.ego_fut_mode, 2)
                            ego_fut_trajs_list.append(outputs_ego_trajs)

                    ego_fut_preds = torch.stack(ego_fut_trajs_list, dim=2)
                else:
                    history_input_output_id.append(input_ids)
                    context_input_ids = torch.cat(history_input_output_id, dim=-1)   
                    output_ids, inputs_embeds, output_embeds, new_input_ids = self.lm_head.generate(
                        inputs=context_input_ids,
                        images=vision_embeded,
                        do_sample=True,
                        temperature=0.1,
                        top_p=0.75,
                        num_beams=1,
                        max_new_tokens=320,
                        use_cache=True
                    )
                    generated_text.append(
                        dict(
                        Q=img_metas[0]['vlm_labels'].data[i],
                        A=self.tokenizer.batch_decode(output_ids, skip_special_tokens=False)[0],
                    ))
                    history_input_output_id.append(output_ids)
                                  
            if self.output_text:
                image_path = Path(img_metas[0]['filename'][0])
                json_directory = image_path.parent.parent.parent.stem 
                save_path = self.save_path
                save_path = save_path + str(json_directory)+'/'
                json_path = f'{image_path.stem}.json'
                json_path_str = str(json_path)
                if not os.path.exists(save_path + json_path_str):
                    mmcv.mkdir_or_exist(save_path)
                with open(save_path + json_path_str, 'w') as file:
                    json.dump(generated_text, file, indent=4)    

            full_match = False
            if not self.qa_pretrain:
                if self.use_gen_token:
                    if self.use_meta_action:
                        speed_value = self.SPT_MAPPING[speed_command]
                        path_value  = self.PA_MAPPING[path_command]
                        cmd_speed = self.command2hot(speed_value, max_dim=7)
                        cmd_path = self.command2hot(path_value, max_dim=6)
                        lat_onehot = torch.tensor(cmd_speed).unsqueeze(0)
                        lon_onehot = torch.tensor(cmd_path).unsqueeze(0)
                        if self.is_decoupling:
                            mask_active_cmd = lat_onehot == 1
                            pw_mask_active_cmd = lon_onehot == 1
                    else:
                        mask_active_cmd = data['ego_fut_cmd'][:,0,0] == 1

                    ego_fut_preds_inactive = ego_fut_preds[~mask_active_cmd].to('cpu')
                    ego_fut_preds = ego_fut_preds[mask_active_cmd].flatten(0,1).to('cpu') # (6, 2)
                    if self.is_decoupling and not self.use_meta_action:
                        pw_ego_fut_preds_inactive = pw_ego_fut_preds[~mask_active_cmd].to('cpu')
                        pw_ego_fut_preds = pw_ego_fut_preds[mask_active_cmd].flatten(0,1).to('cpu') # (6, 2)
                    else:
                        pw_ego_fut_preds_inactive = pw_ego_fut_preds[~pw_mask_active_cmd].to('cpu')
                        pw_ego_fut_preds = pw_ego_fut_preds[pw_mask_active_cmd].flatten(0,1).to('cpu') # (6, 2)
                else:
                    traj = generated_text[0]['A'][0]
                    full_match = re.search(r'\[PT, \((\+?[\d\.-]+, \+?[\d\.-]+)\)(, \(\+?[\d\.-]+, \+?[\d\.-]+\))*\]', traj)

                    if full_match:
                        coordinates_matches = re.findall(r'\(\+?[\d\.-]+, \+?[\d\.-]+\)', full_match.group(0))
                        coordinates = [tuple(map(float, re.findall(r'-?\d+\.\d+', coord))) for coord in coordinates_matches]
                        coordinates_array = np.array(coordinates)
                        ego_fut_preds = torch.tensor(coordinates_array)

            if self.use_gen_token or full_match:
                ego_fut_preds = ego_fut_preds.to(torch.float32) # for fp16 infer
                ego_fut_pred = ego_fut_preds.cumsum(dim=-2) # VAD 输出六个command的轨迹，从和GT一直的command的轨迹中选择计算指标的
                if self.is_decoupling:
                    pw_ego_fut_pred = pw_ego_fut_preds.cumsum(dim=-2)

                if hasattr(self, "planning_memory"):
                    ego_curr_traj = waypoint.view(B,-1,2).detach()
                    ego_curr_traj = ego_curr_traj.cumsum(dim=-2)
                    ego_curr_traj = torch.cat([ego_curr_traj, torch.zeros_like(ego_curr_traj[...,0:1]),torch.ones_like(ego_curr_traj[...,0:1])],dim=-1).unsqueeze(-1).detach()
                    global_curr_traj = torch.matmul(data['ego_pose'].unsqueeze(1).repeat(1,ego_curr_traj.shape[1],1,1),ego_curr_traj)
                    self.planning_memory = global_curr_traj.detach()
                if not (self.fp16_infer or self.fp32_infer) or self.fp16_eval:
                    ego_fut_trajs = data['ego_fut_trajs'][0, 0]
                    ego_fut_trajs = ego_fut_trajs.cumsum(dim=-2)
                    metric_dict_planner_stp3 = self.compute_planner_metric_stp3(
                            pred_ego_fut_trajs = ego_fut_pred[None].to('cpu'),
                            gt_ego_fut_trajs = ego_fut_trajs[None].to('cpu'),
                            gt_agent_boxes = gt_bbox,
                            gt_agent_feats = gt_attr_label.unsqueeze(0),
                            fut_valid_flag = fut_valid_flag # 当前帧是否涵盖6个轨迹
                        )
                    metric_dict.update(metric_dict_planner_stp3)
                    lane_results[0]['fut_valid_flag'] = fut_valid_flag
                    if self.is_decoupling:
                        pw_ego_fut_trajs = data['path_points_future'][0, 0]
                        pw_ego_fut_trajs = pw_ego_fut_trajs.cumsum(dim=-2)
                        metric_dict_planner_stp3 = self.compute_planner_metric_stp3(
                                pred_ego_fut_trajs = pw_ego_fut_pred[None].to('cpu'),
                                gt_ego_fut_trajs = pw_ego_fut_trajs[None].to('cpu'),
                                gt_agent_boxes = gt_bbox,
                                gt_agent_feats = gt_attr_label.unsqueeze(0),
                                fut_valid_flag = fut_valid_flag, # 当前帧是否涵盖6个轨迹4
                                plan_path = True,
                            )
                        pw_metric_dict = {}
                        pw_metric_dict = {f"pw_{k}": v for k, v in metric_dict_planner_stp3.items()}    
                        metric_dict.update(pw_metric_dict)
                        lane_results[0]['fut_valid_flag'] = fut_valid_flag

                else:
                    metric_dict.update({'fut_valid_flag': False})
                    lane_results[0]['fut_valid_flag'] = False
                lane_results[0]['ego_fut_preds'] = torch.nan_to_num(ego_fut_pred)
                lane_results[0]['ego_fut_cmd'] = data['ego_fut_cmd']

                if self.is_decoupling:
                    lane_results[0]['pw_ego_fut_pred'] = torch.nan_to_num(pw_ego_fut_pred)
                if self.use_meta_action:
                    lane_results[0]['speed_value'] = speed_value
                    lane_results[0]['path_value'] = path_value
                    # for Rl

                    if self.rl_training:
                        for k, v in meta_action_info.items():
                            if hasattr(v, "cpu"):
                                meta_action_info[k] = v.cpu()
                            else:
                                meta_action_info[k] = v
                        lane_results[0]['meta_action_info'] = meta_action_info
                        lane_results[0]['ppo_info'] = ppo_info


                if os.getenv('DEBUG_SHOW_PRED', None) == "1":
                    show_dir = os.getenv('DEBUG_SHOW_PRED_DIR', None)
                    ego_fut_trajs_show = None
                    if 'ego_fut_trajs' in data:
                        ego_fut_trajs_show = data['ego_fut_trajs']
                    if not self.use_gen_token:
                        ego_fut_preds_inactive = None
                    if self.is_decoupling and not self.use_meta_action:
                        img_to_show, img_bev, qa_img = self.show_results(data['img'].unsqueeze(0), img_metas, data, ego_fut_trajs=ego_fut_trajs_show,
                            waypoint=ego_fut_preds.unsqueeze(0).unsqueeze(0), waypoint_inactive=ego_fut_preds_inactive, bbox_result=bbox_list, 
                            lane_result=lane_results, metric_dict=metric_dict, use_gt=False, show_dir=show_dir, generated_text=None, pw_waypoint = pw_ego_fut_preds.unsqueeze(0).unsqueeze(0),
                            pw_waypoint_inactive = pw_ego_fut_preds_inactive.unsqueeze(0))
                    elif self.is_decoupling and self.use_meta_action:
                        img_to_show, img_bev, qa_img = self.show_results(data['img'].unsqueeze(0), img_metas, data, ego_fut_trajs=ego_fut_trajs_show,
                            waypoint=ego_fut_preds.unsqueeze(0).unsqueeze(0), waypoint_inactive=ego_fut_preds_inactive, bbox_result=bbox_list, 
                            lane_result=lane_results, metric_dict=metric_dict, use_gt=False, show_dir=show_dir, generated_text=None, pw_waypoint = pw_ego_fut_preds.unsqueeze(0).unsqueeze(0),
                            pw_waypoint_inactive = pw_ego_fut_preds_inactive.unsqueeze(0),meta_action = [speed_command,path_command], action_distribution = action_logits,ref_action_distribution=action_logits)
                    else:
                        img_to_show, img_bev, qa_img = self.show_results(data['img'].unsqueeze(0), img_metas, data, ego_fut_trajs=ego_fut_trajs_show,
                            waypoint=ego_fut_preds.unsqueeze(0).unsqueeze(0), waypoint_inactive=ego_fut_preds_inactive, bbox_result=bbox_list, 
                            lane_result=lane_results, metric_dict=metric_dict, use_gt=False, show_dir=show_dir, generated_text=None, pw_waypoint = None,
                            pw_waypoint_inactive = None, meta_action = commands)
                    lane_results[0]['result_vis'] = dict(img_to_show=img_to_show, img_bev=img_bev, qa_img=qa_img)

            else:
                metric_dict.update({'fut_valid_flag': False})
                lane_results[0]['ego_fut_preds'] = torch.zeros((6, 2), dtype=torch.float32).to(location.device)
                lane_results[0]['ego_fut_cmd'] = data['ego_fut_cmd']
                lane_results[0]['fut_valid_flag'] = fut_valid_flag if not self.qa_pretrain else False
                if self.qa_pretrain:
                    if os.getenv('DEBUG_SHOW_PRED', None) == "1":
                        show_dir = os.getenv('DEBUG_SHOW_PRED_DIR', None)
                        ego_fut_trajs_show = None
                        if 'ego_fut_trajs' in data:
                            ego_fut_trajs_show = data['ego_fut_trajs']
                        img_to_show, img_bev, qa_img = self.show_results(data['img'].unsqueeze(0), img_metas, data, ego_fut_trajs=ego_fut_trajs_show,
                            waypoint=None, waypoint_inactive=None, bbox_result=bbox_list, 
                            lane_result=lane_results, metric_dict=None, use_gt=False, show_dir=show_dir, generated_text=generated_text)
                        lane_results[0]['ss'] = dict(img_to_show=img_to_show, img_bev=img_bev, qa_img=qa_img)

        return bbox_results, generated_text, lane_results, metric_dict
    
    def simple_test(self, img_metas, **data):
        """Test function without augmentaiton."""    
        data['img_feats'] = self.extract_feat(data['img']) 

        bbox_list = [dict() for i in range(len(img_metas))]
        if data['img'].dim() == 4: # (6,3,640,640)
            data['img'] = data['img'].unsqueeze(0)
        bbox_pts, generated_text, lane_results, metric_dict = self.simple_test_pts(
            img_metas, **data)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
            result_dict['metric_results'] = metric_dict
            # print(result_dict['metric_results']['fut_valid_flag']) for debug
        bbox_list[0]['text_out'] = generated_text
        bbox_list[0]['pts_bbox'].update(lane_results[0])
       
        return bbox_list


    def command2hot(self,command,max_dim=6):
        cmd_one_hot = np.zeros(max_dim)
        cmd_one_hot[command] = 1
        return cmd_one_hot

    def compute_planner_metric_stp3(
        self,
        pred_ego_fut_trajs,
        gt_ego_fut_trajs,
        gt_agent_boxes,
        gt_agent_feats,
        fut_valid_flag,
        plan_path = False,
    ):
        """Compute planner metric for one sample same as stp3."""
        if not plan_path:
            metric_dict = {
                'plan_L2_1s':0,
                'plan_L2_2s':0,
                'plan_L2_3s':0,
                'plan_obj_col_1s':0,
                'plan_obj_col_2s':0,
                'plan_obj_col_3s':0,
                'plan_obj_box_col_1s':0,
                'plan_obj_box_col_2s':0,
                'plan_obj_box_col_3s':0,
            }
        else:
            metric_dict = {
                'plan_L2_1':0,
                'plan_L2_2':0,
                'plan_L2_3':0,
                'plan_L2_4':0,
                'plan_L2_5':0,
            }
        metric_dict['fut_valid_flag'] = fut_valid_flag
        if plan_path:
            future_second = 5
        else:
            future_second = 3
        assert pred_ego_fut_trajs.shape[0] == 1, 'only support bs=1'
        if self.planning_metric is None:
            self.planning_metric = PlanningMetric()
        segmentation, pedestrian = self.planning_metric.get_label(
            gt_agent_boxes, gt_agent_feats)
        occupancy = torch.logical_or(segmentation, pedestrian)

        for i in range(future_second):
            if fut_valid_flag or pred_ego_fut_trajs.size(1)==6 :
                cur_time = (i+1)*2
                traj_L2 = self.planning_metric.compute_L2(
                    pred_ego_fut_trajs[0, :cur_time].detach().to(gt_ego_fut_trajs.device),
                    gt_ego_fut_trajs[0, :cur_time]
                )
                if not plan_path:
                    obj_coll, obj_box_coll = self.planning_metric.evaluate_coll(
                        pred_ego_fut_trajs[:, :cur_time].detach(),
                        gt_ego_fut_trajs[:, :cur_time],
                        occupancy)
                    metric_dict['plan_obj_col_{}s'.format(i+1)] = np.nan_to_num(obj_coll.mean().item())
                    metric_dict['plan_obj_box_col_{}s'.format(i+1)] = np.nan_to_num(obj_box_coll.mean().item())
                    metric_dict['plan_L2_{}s'.format(i+1)] = np.nan_to_num(traj_L2)
                else:
                    metric_dict['plan_L2_{}'.format(i+1)] = np.nan_to_num(traj_L2)
            else:
                if not plan_path:
                    metric_dict['plan_L2_{}s'.format(i+1)] = 0.0
                    metric_dict['plan_obj_col_{}s'.format(i+1)] = 0.0
                    metric_dict['plan_obj_box_col_{}s'.format(i+1)] = 0.0
                else:
                    metric_dict['plan_L2_{}'.format(i+1)] = 0.0
            
        return metric_dict
    def assign_pred_to_gt_vip3d(
        self,
        bbox_result,
        gt_bbox,
        gt_label,
        match_dis_thresh=2.0
    ):
        """Assign pred boxs to gt boxs according to object center preds in lcf.
        Args:
            bbox_result (dict): Predictions.
                'boxes_3d': (LiDARInstance3DBoxes)
                'scores_3d': (Tensor), [num_pred_bbox]
                'labels_3d': (Tensor), [num_pred_bbox]
                'trajs_3d': (Tensor), [fut_ts*2]
            gt_bboxs (LiDARInstance3DBoxes): GT Bboxs.
            gt_label (Tensor): GT labels for gt_bbox, [num_gt_bbox].
            match_dis_thresh (float): dis thresh for determine a positive sample for a gt bbox.

        Returns:
            matched_bbox_result (np.array): assigned pred index for each gt box [num_gt_bbox].
        """     
        dynamic_list = [0,1,3,4,6,7,8]
        matched_bbox_result = torch.ones(
            (len(gt_bbox)), dtype=torch.long) * -1  # -1: not assigned
        gt_centers = gt_bbox.center[:, :2]
        pred_centers = bbox_result['boxes_3d'].center[:, :2]
        dist = torch.linalg.norm(pred_centers[:, None, :] - gt_centers[None, :, :], dim=-1)
        pred_not_dyn = [label not in dynamic_list for label in bbox_result['labels_3d']]
        gt_not_dyn = [label not in dynamic_list for label in gt_label]
        dist[pred_not_dyn] = 1e6
        dist[:, gt_not_dyn] = 1e6
        dist[dist > match_dis_thresh] = 1e6

        r_list, c_list = linear_sum_assignment(dist)

        for i in range(len(r_list)):
            if dist[r_list[i], c_list[i]] <= match_dis_thresh:
                matched_bbox_result[c_list[i]] = r_list[i]

        return matched_bbox_result
    
    def compute_motion_metric_vip3d(
        self,
        gt_bbox,
        gt_label,
        gt_attr_label,
        pred_bbox,
        matched_bbox_result,
        mapped_class_names,
        match_dis_thresh=2.0,
    ):
        """Compute EPA metric for one sample.
        Args:
            gt_bboxs (LiDARInstance3DBoxes): GT Bboxs.
            gt_label (Tensor): GT labels for gt_bbox, [num_gt_bbox].
            pred_bbox (dict): Predictions.
                'boxes_3d': (LiDARInstance3DBoxes)
                'scores_3d': (Tensor), [num_pred_bbox]
                'labels_3d': (Tensor), [num_pred_bbox]
                'trajs_3d': (Tensor), [fut_ts*2]
            matched_bbox_result (np.array): assigned pred index for each gt box [num_gt_bbox].
            match_dis_thresh (float): dis thresh for determine a positive sample for a gt bbox.

        Returns:
            EPA_dict (dict): EPA metric dict of each cared class.
        """
        motion_cls_names = ['car', 'pedestrian']
        motion_metric_names = ['gt', 'cnt_ade', 'cnt_fde', 'hit',
                               'fp', 'ADE', 'FDE', 'MR']
        
        metric_dict = {}
        for met in motion_metric_names:
            for cls in motion_cls_names:
                metric_dict[met+'_'+cls] = 0.0

        veh_list = [0,1,3,4]
        ignore_list = ['construction_vehicle', 'barrier',
                       'traffic_cone', 'motorcycle', 'bicycle']

        for i in range(pred_bbox['labels_3d'].shape[0]):
            pred_bbox['labels_3d'][i] = 0 if pred_bbox['labels_3d'][i] in veh_list else pred_bbox['labels_3d'][i]
            box_name = mapped_class_names[pred_bbox['labels_3d'][i]]
            if box_name in ignore_list:
                continue
            if i not in matched_bbox_result:
                metric_dict['fp_'+box_name] += 1

        for i in range(gt_label.shape[0]):
            gt_label[i] = 0 if gt_label[i] in veh_list else gt_label[i]
            box_name = mapped_class_names[gt_label[i]]
            if box_name in ignore_list:
                continue
            gt_fut_masks = gt_attr_label[i][self.fut_ts*2:self.fut_ts*3]
            num_valid_ts = sum(gt_fut_masks==1)
            if num_valid_ts == self.fut_ts:
                metric_dict['gt_'+box_name] += 1
            if matched_bbox_result[i] >= 0 and num_valid_ts > 0:
                metric_dict['cnt_ade_'+box_name] += 1
                m_pred_idx = matched_bbox_result[i]
                gt_fut_trajs = gt_attr_label[i][:self.fut_ts*2].reshape(-1, 2)
                gt_fut_trajs = gt_fut_trajs[:num_valid_ts]
                self.fut_mode = 1
                pred_fut_trajs = pred_bbox['trajs_3d'][m_pred_idx].reshape(self.fut_mode, self.fut_ts, 2)
                pred_fut_trajs = pred_fut_trajs[:, :num_valid_ts, :]
                gt_fut_trajs = gt_fut_trajs.cumsum(dim=-2)
                pred_fut_trajs = pred_fut_trajs.cumsum(dim=-2)
                gt_fut_trajs = gt_fut_trajs + gt_bbox[i].center[0, :2]
                pred_fut_trajs = pred_fut_trajs + pred_bbox['boxes_3d'][int(m_pred_idx)].center[0, :2]

                dist = torch.linalg.norm(gt_fut_trajs[None, :, :] - pred_fut_trajs, dim=-1)
                ade = dist.sum(-1) / num_valid_ts
                ade = ade.min()

                metric_dict['ADE_'+box_name] += ade
                if num_valid_ts == self.fut_ts:
                    fde = dist[:, -1].min()
                    metric_dict['cnt_fde_'+box_name] += 1
                    metric_dict['FDE_'+box_name] += fde
                    if fde <= match_dis_thresh:
                        metric_dict['hit_'+box_name] += 1
                    else:
                        metric_dict['MR_'+box_name] += 1

        return metric_dict

    def show_results(self, img, img_metas, data, map_gt_bboxes_3d=None,
                          map_gt_labels_3d=None, ego_fut_trajs=None, waypoint_inactive=None,
                          outs_bbox=None, outs_lane=None, bbox_result=None, lane_result=None,
                          waypoint=None, metric_dict=None, use_gt=False, show_dir=None, generated_text=None,pw_waypoint = None,pw_waypoint_inactive=None,meta_action = None,
                          action_distribution = None, ref_action_distribution = None,):
        tasks = [
            dict(
                task_name = 'class',
                num_out = 9,
                level = 'box',
                names =[
                    'car','van','truck','bicycle','traffic_sign','traffic_cone','traffic_light','pedestrian','others'
                ]
            ),
        ]

        grid_config = {
            'xbound': [-50.0, 50.0, 0.5],
            'ybound': [-50.0, 50.0, 0.5],
            'zbound': [-10.0, 10.0, 20.0],
            'dbound': [1.0, 64.0, 1.0]}

        cams = ['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT',
                'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
        cams_canvas_seq = [1, 0, 2, 4, 3, 5]
        intrinsics = data['cam_intrinsic'][0].cpu().numpy()
        # cam2lidar = data['extrinsics'][0].cpu().numpy()
        lidar2img = data['lidar2img'][0].cpu().numpy()
        lidar2cam = np.linalg.inv(intrinsics) @ lidar2img
        # lidar2cam = np.array(img_metas[0]['ori_extrinsics'])
        # intrinsics = np.array(img_metas[0]['ori_intrinsics'])[:, :3, :3]
        cam2lidar = np.linalg.inv(lidar2cam)

        intrinsics = intrinsics[:, :3, :3]
        all_rots = cam2lidar[:, :3, :3]
        all_trans = cam2lidar[:, :3, 3]

        # filenames = img_metas[0]['filename']
        # filenames = {name.split('/')[4] : name for name in filenames}
        sample_idx = img_metas[0]['scene_token'] + '/frame_idx_' + str(img_metas[0]['frame_idx'])
        if not use_gt:
            if bbox_result is None:
                bbox_result = self.pts_bbox_head.get_bboxes(outs_bbox, img_metas)
            if lane_result is None:
                lane_result = self.map_head.get_bboxes(outs_lane, img_metas)

        aug_imgs = img[0, 0].permute(0, 2, 3, 1).cpu().numpy()  # [6, h, w, 3] #NOTE@Jianfeng: bs 1 num frame 1
        norm_mean = [0.485, 0.456, 0.406]
        norm_std = [0.229, 0.224, 0.225]
        norm_mean = np.array(norm_mean).reshape((1, 1, 1, 3))
        norm_std = np.array(norm_std).reshape((1, 1, 1, 3))
        aug_imgs = aug_imgs[..., :3] * norm_std + norm_mean
        aug_imgs = aug_imgs * 255
        aug_imgs = aug_imgs[..., [2, 1, 0]] # cv2 write uses BGR image 
        aug_imgs = aug_imgs.astype(np.uint8)
        aug_imgs = np.ascontiguousarray(aug_imgs)

        all_imgs = []
        rots = []
        trans = []
        intrins = []
        for img_i in cams_canvas_seq:
            cam_name = cams[img_i]
            # index = list(filenames.keys()).index(cam_name)
            index = img_i
            # filename = filenames[cam_name]
            # img = cv2.imread(filename)
            # img = np.array(img_metas[0]['ori_imgs'][index])
            img = aug_imgs[index]
            all_imgs.append(img)
            rots.append(all_rots[index])
            trans.append(all_trans[index])
            intrins.append(intrinsics[index])

        if not use_gt:
            # mask = result[0][1].detach().cpu().numpy() > 0.2
            # bbox = result[0][0].detach().cpu().numpy()
            bbox = format_bbox(bbox_result[0][0]).copy()
            # NOTE@Jianfeng: bench2drive coords -> vcs coords
            bbox[:, [3, 4, 5]] = bbox[:, [4, 3, 5]]
            # bbox[:, 6] -= torch.pi / 2
            bbox[:, 6] = - (bbox[:, 6] + torch.pi / 2)
            bbox[:, 2] += bbox[:, 5] / 2 # center -> gravity center
            pred_dict = {
                'box' : bbox,
                'class' : bbox_result[0][2].detach().cpu().numpy(),
                'score' : bbox_result[0][1].detach().cpu().numpy(),
            }
        else:
            bbox = format_bbox(img_metas[0]['gt_bboxes_3d']).copy()
            # NOTE@Jianfeng: bench2drive coords -> vcs coords
            bbox[:, [3, 4, 5]] = bbox[:, [4, 3, 5]]
            # bbox[:, 6] -= torch.pi / 2
            bbox[:, 6] = - (bbox[:, 6] + torch.pi / 2)
            bbox[:, 2] += bbox[:, 5] / 2 # center -> gravity center
            pred_dict = {
                'box' : bbox,
                'class' : format_bbox(img_metas[0]['gt_labels_3d'])
            }

        if show_dir is not None:
            os.makedirs(show_dir, exist_ok=True)

        img_to_show, img_bev, ratio = show_multicam_bboxes(all_imgs, intrins, rots, trans, cams,
                             grid_config, 1, tasks, pred_dict,
                             dataset_type='b2d')

        lidar2ego = img_metas[0]['lidar2ego']
        if not use_gt:
            lane_pts_3d = lane_result[0]['map_pts_3d'].detach().cpu().numpy()
            lane_pts_3d[..., 2] = -lidar2ego[2, 3]
            lane_labels_3d = lane_result[0]['map_labels_3d'].detach().cpu().numpy()
        else:
            lane_pts_3d = map_gt_bboxes_3d[0].fixed_num_sampled_points.detach().cpu().numpy()
            lane_pts_3d = np.concatenate(
                (lane_pts_3d, -np.ones_like(lane_pts_3d[:, :, 0:1]) * lidar2ego[2, 3],), axis=-1,)
            lane_labels_3d = map_gt_labels_3d[0].detach().cpu().numpy()
        ld_infos = {
            "lane_pts_3d": lane_pts_3d,
            "lane_labels_3d": lane_labels_3d,
            "class_names": ['Broken','Solid','SolidSolid','Center','TrafficLight','StopSign'],
            "grid_config": grid_config,
        }
        calib_infos = {
            "intrins": intrins,
            "rots": rots,
            "trans": trans,
            # "post_rots": post_rots,
            # "post_trans": post_trans,
        }
        img_to_show, img_bev = draw_ld_vis(
            all_imgs,
            calib_infos,
            cams,
            img_bev=img_bev,
            ratio=ratio,
            ld_infos=ld_infos,
            draw_aug_img=False,
            aug_imgs=aug_imgs,
        )
        if meta_action is not None:
            img_bev = cv2.putText(img_bev, meta_action[0], (20, 40), cv2.FONT_HERSHEY_TRIPLEX , 0.8, 1, 1, cv2.LINE_AA)
            img_bev = cv2.putText(img_bev, meta_action[1], (20, 60), cv2.FONT_HERSHEY_TRIPLEX , 0.8, 1, 1, cv2.LINE_AA)
        else:
            commands = ["LEFT", "RIGHT", "STRAIGHT", "LANE FOLLOW", "CHANGE LANE LEFT", "CHANGE LANE RIGHT"]
            command = commands[int(data['command'].item())]
            img_bev = cv2.putText(img_bev, command, (20, 40), cv2.FONT_HERSHEY_TRIPLEX , 0.8, 1, 1, cv2.LINE_AA)

        if pw_waypoint is not None:
            pw_all_imgs = [img.copy() for img in all_imgs]
            pw_img_bev = img_bev.copy()
            pw_aug_imgs = aug_imgs.copy()
        # cv2.FONT_HERSHEY_SIMPLEX
        # if metric_dict is not None and 'plan_L2_2s' in metric_dict and 'fut_valid_flag' in metric_dict:
        #     metric_text = f"plan_L2_2s: {metric_dict['plan_L2_2s']:.3f}" 
        #     img_bev = cv2.putText(img_bev, metric_text, (20, 60), cv2.FONT_HERSHEY_TRIPLEX, 0.4, 1, 1, cv2.LINE_AA)
        #     metric_text = f"fut_valid_flag: {metric_dict['fut_valid_flag']}" 
        #     img_bev = cv2.putText(img_bev, metric_text, (20, 80), cv2.FONT_HERSHEY_TRIPLEX, 0.4, 1, 1, cv2.LINE_AA)

        if waypoint_inactive is not None:
            if len(waypoint_inactive.shape) < 4:
                waypoint_inactive = waypoint_inactive.unsqueeze(0)
            for i in range(waypoint_inactive.shape[1]):
                waypoint_pts = torch.nan_to_num(waypoint_inactive).cumsum(dim=-2) # sample 0, command i
                waypoint_pts = waypoint_pts[0, i:i+1].to(torch.float32).detach().cpu().numpy()
                waypoint_pts = np.concatenate(
                    (waypoint_pts, -np.ones_like(waypoint_pts[:, :, 0:1]) * lidar2ego[2, 3],), axis=-1,)
                waypoint_infos = {
                    "lane_pts_3d": waypoint_pts,
                    "lane_labels_3d": torch.zeros((1,)).to(torch.int).detach().cpu().numpy(),
                    "class_names": ['waypointInactive'],
                    "grid_config": grid_config,
                }
                img_to_show, img_bev = draw_ld_vis(
                    all_imgs,
                    calib_infos,
                    cams,
                    img_bev=img_bev,
                    ratio=ratio,
                    ld_infos=waypoint_infos,
                    draw_aug_img=False,
                    aug_imgs=aug_imgs,
                )
            
        if pw_waypoint_inactive is not None:
            if len(pw_waypoint_inactive.shape) < 4:
                pw_waypoint_inactive = pw_waypoint_inactive.unsqueeze(0)
            for i in range(pw_waypoint_inactive.shape[1]):
                pw_waypoint_pts = torch.nan_to_num(pw_waypoint_inactive).cumsum(dim=-2)
                waypoint_pts = pw_waypoint_pts[0, i:i+1].to(torch.float32).detach().cpu().numpy()
                waypoint_pts = np.concatenate(
                    (waypoint_pts, -np.ones_like(waypoint_pts[:, :, 0:1]) * lidar2ego[2, 3],), axis=-1,)
                waypoint_infos = {
                    "lane_pts_3d": waypoint_pts,
                    "lane_labels_3d": torch.zeros((1,)).to(torch.int).detach().cpu().numpy(),
                    "class_names": ['pwwaypointInactive'],
                    "grid_config": grid_config,
                }
                pw_img_to_show, pw_img_bev = draw_ld_vis(
                    pw_all_imgs,
                    calib_infos,
                    cams,
                    img_bev=pw_img_bev,
                    ratio=ratio,
                    ld_infos=waypoint_infos,
                    draw_aug_img=False,
                    aug_imgs=pw_aug_imgs,
                )
        # if ego_fut_trajs is not None:
        #     ego_fut_trajs_pts = ego_fut_trajs.cumsum(dim=-2)
        #     ego_fut_trajs_pts = ego_fut_trajs_pts[0].to(torch.float32).detach().cpu().numpy()
        #     ego_fut_trajs_pts = np.concatenate(
        #         (ego_fut_trajs_pts, -np.ones_like(ego_fut_trajs_pts[:, :, 0:1]) * lidar2ego[2, 3],), axis=-1,)
        #     waypoint_infos = {
        #         "lane_pts_3d": ego_fut_trajs_pts,
        #         "lane_labels_3d": torch.zeros((1,)).to(torch.int).detach().cpu().numpy(),
        #         "class_names": ['waypointGT'],
        #         "grid_config": grid_config,
        #     }
        #     img_to_show, img_bev = draw_ld_vis(
        #         all_imgs,
        #         calib_infos,
        #         cams,
        #         img_bev=img_bev,
        #         ratio=ratio,
        #         ld_infos=waypoint_infos,
        #         draw_aug_img=False,
        #         aug_imgs=aug_imgs,
        #     )

        if waypoint is not None:

            waypoint_pts = torch.nan_to_num(waypoint)
            waypoint_pts = waypoint_pts[0].to(torch.float32).detach().cpu().numpy()
            waypoint_pts = np.concatenate(
                (waypoint_pts, -np.ones_like(waypoint_pts[:, :, 0:1]) * lidar2ego[2, 3],), axis=-1,)
            waypoint_infos = {
                "lane_pts_3d": waypoint_pts,
                "lane_labels_3d": torch.zeros((1,)).to(torch.int).detach().cpu().numpy(),
                "class_names": ['waypoint'],
                "grid_config": grid_config,
            }
            img_to_show, img_bev = draw_ld_vis(
                all_imgs,
                calib_infos,
                cams,
                img_bev=img_bev,
                ratio=ratio,
                ld_infos=waypoint_infos,
                draw_aug_img=False,
                aug_imgs=aug_imgs,
            )

        if pw_waypoint is not None:
            pw_waypoint_pts = torch.nan_to_num(pw_waypoint)
            pw_waypoint_pts = pw_waypoint_pts[0].to(torch.float32).detach().cpu().numpy()
            pw_waypoint_pts = np.concatenate(
                (pw_waypoint_pts, -np.ones_like(pw_waypoint_pts[:, :, 0:1]) * lidar2ego[2, 3],), axis=-1,)
            pw_waypoint_infos = {
                "lane_pts_3d": pw_waypoint_pts,
                "lane_labels_3d": torch.zeros((1,)).to(torch.int).detach().cpu().numpy(),
                "class_names": ['pw_waypoint'],
                "grid_config": grid_config,
            }
            pw_img_to_show, pw_img_bev = draw_ld_vis(
                pw_all_imgs,
                calib_infos,
                cams,
                img_bev=pw_img_bev,
                ratio=ratio,
                ld_infos=pw_waypoint_infos,
                draw_aug_img=False,
                aug_imgs=pw_aug_imgs,
            )
            pw_img = np.concatenate([pw_img_to_show, pw_img_bev], axis=1)
        
        img = np.concatenate([img_to_show, img_bev], axis=1)
        qa_img = np.ones([img.shape[0]//2, img.shape[1],3],dtype=np.uint8)*255
        if generated_text is not None and len(generated_text):
            scene_desc_list = []
            for qa in generated_text:
                question = 'Q: ' + qa['Q']
                answer = 'A: ' + qa['A'][0]
                for i in range(0,len(question),200):
                    scene_desc_list.append(question[i:i+200])
                for i in range(0,len(answer),200):
                    scene_desc_list.append(answer[i:i+200])
            y = 15
            for desc in scene_desc_list:
                qa_img = cv2.putText(qa_img, desc, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1, 1, cv2.LINE_AA)
                y += 20
            img = np.concatenate([img, qa_img],axis=0)

        if action_distribution is not None:
            action_distribution = torch.exp(action_distribution)
            speed_commands = {
                0: 'mms',
                1: 's',
                2: 'mss',
                3: 'su',
                4: 'sd',
                5: 'mfs',
                6: 'sdr'
            }
            plt.figure(figsize=(2, 2))
            plt.bar(range(len(action_distribution[0])), action_distribution[0].to('cpu').numpy(), color='skyblue')
            plt.xticks(range(len(action_distribution[0])), list(speed_commands.values()), rotation=45, ha='right')
            plt.ylim(0, 1)
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close()
            plt_img = cv2.imdecode(np.frombuffer(buf.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR)
        if ref_action_distribution is not None:
            ref_action_distribution = torch.exp(ref_action_distribution)
            speed_commands = {
                0: 'mms',
                1: 'r_s',
                2: 'mss',
                3: 'su',
                4: 'sd',
                5: 'mfs',
                6: 'sdr'
            }
            plt.figure(figsize=(2, 2))
            plt.bar(range(len(ref_action_distribution[0])), ref_action_distribution[0].to('cpu').numpy(), color='skyblue')
            plt.xticks(range(len(ref_action_distribution[0])), list(speed_commands.values()), rotation=45, ha='right')
            plt.ylim(0, 1)
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close()
            ref_plt_img = cv2.imdecode(np.frombuffer(buf.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR)
            plt_img = np.concatenate([plt_img, ref_plt_img], axis=1)
        if self.rl_training:
            h,w,_ = plt_img.shape
            pw_img_to_show[-h:, -w:] = plt_img
        if self.is_decoupling and self.rl_training:
            img_to_show = np.concatenate([img_to_show, pw_img_to_show], axis=0)
            img_bev = np.concatenate([img_bev, pw_img_bev], axis=0)

        if show_dir is not None:
            # 确定文件名
            if use_gt:
                file_name = 'vis_' + sample_idx.replace('/', '_') + '_gt.jpg'
                output_path = os.path.join(show_dir, file_name)
            else:
                file_name = 'vis_' + sample_idx.replace('/', '_') + '.jpg'
                output_path = os.path.join(show_dir, file_name)
            if pw_waypoint is not None:
                pw_file_name = 'vis_path_' + sample_idx.replace('/', '_') + '.jpg'
                pw_output_path = os.path.join(show_dir, pw_file_name)
            text = file_name
            position = (10, 30)  
            font = cv2.FONT_HERSHEY_SIMPLEX  
            font_scale = 0.7 
            color = (255, 255, 255)  
            thickness = 2 
            cv2.putText(img, text, position, font, font_scale, color, thickness)
            if self.is_decoupling:
                img = np.concatenate([img, pw_img],axis=0)

            cv2.imwrite(output_path, img)

        return img_to_show, img_bev, qa_img

    def future_states_predict(self, batch_size, sample, hidden_states, current_states):

        future_prediction_input = sample.unsqueeze(0).expand(self.fut_ts, -1, -1, -1) # (6, 1, 32, 1)
        #
        # future_states = self.future_prediction(future_prediction_input, hidden_state)
        future_prediction_input = future_prediction_input.reshape(self.fut_ts, -1, self.latent_dim) # (6, 1, 32)

        hidden_states = hidden_states.permute(1,0,2)
        if self.lm_model_type == 'llama_v3':
            hidden_state = hidden_states.reshape(self.layer_dim, -1, int(2048/4))
        elif self.lm_model_type == 'qwen2':
            hidden_state = hidden_states.reshape(self.layer_dim, -1, int(896/4))
        elif self.lm_model_type == 'qwen25_3B':
            hidden_state = hidden_states.reshape(self.layer_dim, -1, int(2048/4))
        else:
            hidden_state = hidden_states.reshape(self.layer_dim, -1, int(4096/4))
        # future_states, future_hidden = self.state_gru(future_prediction_input, hidden_state)
        future_states = self.predict_model(future_prediction_input, hidden_state.contiguous()) # (6, 1, 4096)

        current_states_hs = current_states.unsqueeze(0).repeat(6, 1, 1, 1)
        future_states_hs = future_states.reshape(self.fut_ts, batch_size, -1, future_states.shape[2]) # （6，1，1，4096）

        if self.with_cur:
            states_hs = torch.cat((current_states_hs, future_states_hs), dim=-1)
        else:
            states_hs = future_states_hs

        return states_hs, future_states_hs
    
    def pw_future_states_predict(self, batch_size, sample, hidden_states, current_states):

        future_prediction_input = sample.unsqueeze(0).expand(self.fut_ps, -1, -1, -1) # (6, 1, 32, 1)
        #
        # future_states = self.future_prediction(future_prediction_input, hidden_state)
        future_prediction_input = future_prediction_input.reshape(self.fut_ps, -1, self.latent_dim) # (6, 1, 32)

        hidden_states = hidden_states.permute(1,0,2)
        if self.lm_model_type == 'llama_v3':
            hidden_state = hidden_states.reshape(self.layer_dim, -1, int(2048/4))
        elif self.lm_model_type == 'qwen2':
            hidden_state = hidden_states.reshape(self.layer_dim, -1, int(896/4))
        elif self.lm_model_type == 'qwen25_3B':
            hidden_state = hidden_states.reshape(self.layer_dim, -1, int(2048/4))
        else:
            hidden_state = hidden_states.reshape(self.layer_dim, -1, int(4096/4))
        # future_states, future_hidden = self.state_gru(future_prediction_input, hidden_state)
        future_states = self.pw_predict_model(future_prediction_input, hidden_state.contiguous()) # (6, 1, 4096)

        current_states_hs = current_states.unsqueeze(0).repeat(self.fut_ps, 1, 1, 1)
        future_states_hs = future_states.reshape(self.fut_ps, batch_size, -1, future_states.shape[2]) # （6，1，1，4096）

        if self.with_cur:
            states_hs = torch.cat((current_states_hs, future_states_hs), dim=-1)
        else:
            states_hs = future_states_hs

        return states_hs, future_states_hs
    def get_future_labels(self, gt_labels_3d, gt_attr_labels, ego_fut_trajs, device):

        agent_dim = 300
        veh_list = [0, 1, 2]
        # veh_list = [0, 1, 3, 4]
        # mapped_class_names = [
        #     'car', 'truck', 'construction_vehicle', 'bus',
        #     'trailer', 'barrier', 'motorcycle', 'bicycle',
        #     'pedestrian', 'traffic_cone'
        # ]
        mapped_class_names = [
            'car','van','truck','bicycle','traffic_sign','traffic_cone','traffic_light','pedestrian','others'
        ]
        # ignore_list = ['construction_vehicle', 'barrier',
        #                'traffic_cone', 'motorcycle', 'bicycle']
        ignore_list = []

        batch_size = len(gt_labels_3d)

        # gt_label = gt_labels_3d[0]
        # gt_attr_label = gt_attr_labels[0]

        gt_fut_trajs_bz_list = []

        for bz in range(batch_size):
            gt_fut_trajs_list = []
            gt_label = gt_labels_3d[bz]
            gt_attr_label = gt_attr_labels[bz]
            for i in range(gt_label.shape[0]):
                gt_label[i] = 0 if gt_label[i] in veh_list else gt_label[i]
                box_name = mapped_class_names[gt_label[i]]
                if box_name in ignore_list:
                    continue
                gt_fut_masks = gt_attr_label[i][self.fut_ts * 2:self.fut_ts * 3]
                num_valid_ts = sum(gt_fut_masks == 1)
                gt_fut_traj = gt_attr_label[i][:self.fut_ts * 2].reshape(-1, 2)
                gt_fut_traj = gt_fut_traj[:num_valid_ts]
                if gt_fut_traj.shape[0] == 0:
                    gt_fut_traj = torch.zeros([self.fut_ts - gt_fut_traj.shape[0], 2], device=device)
                if gt_fut_traj.shape[0] < self.fut_ts:
                    gt_fut_traj = torch.cat(
                        (gt_fut_traj, torch.zeros([self.fut_ts - gt_fut_traj.shape[0], 2], device=device)), 0)
                gt_fut_trajs_list.append(gt_fut_traj)

            if len(gt_fut_trajs_list) != 0 & len(gt_fut_trajs_list) < agent_dim:
                gt_fut_trajs = torch.cat(
                    (torch.stack(gt_fut_trajs_list),
                     torch.zeros([agent_dim - len(gt_fut_trajs_list), self.fut_ts, 2], device=device)), 0)
            else:
                gt_fut_trajs = torch.zeros([agent_dim, self.fut_ts, 2], device=device)

            gt_fut_trajs_bz_list.append(gt_fut_trajs)

        if len(gt_fut_trajs_bz_list) != 0:
            gt_trajs = torch.cat((torch.stack(gt_fut_trajs_bz_list).repeat(1, 6, 1, 1), ego_fut_trajs), dim=1)
        else:
            gt_trajs = ego_fut_trajs
        # future_states =  gt_trajs.reshape(batch_size, gt_trajs.shape[1], -1)

        # [bz, a, t, 2]
        return gt_trajs.reshape(batch_size, gt_trajs.shape[1], -1)

    def distribution_forward(self, present_features, future_distribution_inputs=None, noise=None):
        """
        Parameters
        ----------
            present_features: 5-D output from dynamics module with shape (b, 1, c, h, w)
            future_distribution_inputs: 5-D tensor containing labels shape (b, s, cfg.PROB_FUTURE_DIM, h, w)
            noise: a sample from a (0, 1) gaussian with shape (b, s, latent_dim). If None, will sample in function

        Returns
        -------
            sample: sample taken from present/future distribution, broadcast to shape (b, s, latent_dim, h, w)
            present_distribution_mu: shape (b, s, latent_dim)
            present_distribution_log_sigma: shape (b, s, latent_dim)
            future_distribution_mu: shape (b, s, latent_dim)
            future_distribution_log_sigma: shape (b, s, latent_dim)
        """

        b = present_features.shape[0]
        c = present_features.shape[1]
        present_mu, present_log_sigma = self.present_distribution(present_features) # (1, 1, 32)

        future_mu, future_log_sigma = None, None
        if future_distribution_inputs is not None:
            # Concatenate future labels to z_t
            # future_features = future_distribution_inputs[:, 1:].contiguous().view(b, 1, -1, h, w)
            future_features = torch.cat([present_features, future_distribution_inputs], dim=2)
            future_mu, future_log_sigma = self.future_distribution(future_features)

        if noise is None:
            if self.training:
                noise = torch.randn_like(present_mu)
            else:
                noise = torch.randn_like(present_mu)
        # print('################################')
        # print('noise: ', noise)
        # print('################################')
        if self.training:
            mu = future_mu
            sigma = torch.exp(future_log_sigma)
        else:
            mu = present_mu
            sigma = torch.exp(present_log_sigma)
        sample = mu + sigma * noise # （1, 1, 32）

        # Spatially broadcast sample to the dimensions of present_features
        sample = sample.permute(0, 2, 1).expand(b, self.latent_dim, c)
        
        output_distribution = {
            'present_mu': present_mu,
            'present_log_sigma': present_log_sigma,
            'future_mu': future_mu,
            'future_log_sigma': future_log_sigma,
        }

        return sample, output_distribution

    def pw_distribution_forward(self, present_features, future_distribution_inputs=None, noise=None):
        """
        Parameters
        ----------
            present_features: 5-D output from dynamics module with shape (b, 1, c, h, w)
            future_distribution_inputs: 5-D tensor containing labels shape (b, s, cfg.PROB_FUTURE_DIM, h, w)
            noise: a sample from a (0, 1) gaussian with shape (b, s, latent_dim). If None, will sample in function

        Returns
        -------
            sample: sample taken from present/future distribution, broadcast to shape (b, s, latent_dim, h, w)
            present_distribution_mu: shape (b, s, latent_dim)
            present_distribution_log_sigma: shape (b, s, latent_dim)
            future_distribution_mu: shape (b, s, latent_dim)
            future_distribution_log_sigma: shape (b, s, latent_dim)
        """

        b = present_features.shape[0]
        c = present_features.shape[1]
        present_mu, present_log_sigma = self.pw_present_distribution(present_features) # (1, 1, 32)

        future_mu, future_log_sigma = None, None
        if future_distribution_inputs is not None:
            # Concatenate future labels to z_t
            # future_features = future_distribution_inputs[:, 1:].contiguous().view(b, 1, -1, h, w)
            future_features = torch.cat([present_features, future_distribution_inputs], dim=2)
            future_mu, future_log_sigma = self.pw_future_distribution(future_features)

        if noise is None:
            if self.training:
                noise = torch.randn_like(present_mu)
            else:
                noise = torch.randn_like(present_mu)
        # print('################################')
        # print('noise: ', noise)
        # print('################################')
        if self.training:
            mu = future_mu
            sigma = torch.exp(future_log_sigma)
        else:
            mu = present_mu
            sigma = torch.exp(present_log_sigma)
        sample = mu + sigma * noise # （1, 1, 32）

        # Spatially broadcast sample to the dimensions of present_features
        sample = sample.permute(0, 2, 1).expand(b, self.latent_dim, c)
        
        output_distribution = {
            'present_mu': present_mu,
            'present_log_sigma': present_log_sigma,
            'future_mu': future_mu,
            'future_log_sigma': future_log_sigma,
        }

        return sample, output_distribution

    def loss_planning(self,
                      ego_fut_preds,
                      ego_fut_gt,
                      ego_fut_masks,
                      ego_fut_cmd,
                      lane_preds = None,
                      lane_score_preds = None,
                      agent_preds= None,
                      agent_fut_preds= None,
                      agent_score_preds= None,
                      agent_fut_cls_preds= None,
                      path_loss = False,
                      ):
        """"Loss function for ego vehicle planning.
        Args:
            ego_fut_preds (Tensor): [B, ego_fut_mode, fut_ts, 2]
            ego_fut_gt (Tensor): [B, fut_ts, 2]
            ego_fut_masks (Tensor): [B, fut_ts]
            ego_fut_cmd (Tensor): [B, ego_fut_mode]
            lane_preds (Tensor): [B, num_vec, num_pts, 2]
            lane_score_preds (Tensor): [B, num_vec, 3]
            agent_preds (Tensor): [B, num_agent, 2]
            agent_fut_preds (Tensor): [B, num_agent, fut_mode, fut_ts, 2]
            agent_score_preds (Tensor): [B, num_agent, 10]
            agent_fut_cls_scores (Tensor): [B, num_agent, fut_mode]
        Returns:
            loss_plan_reg (Tensor): planning reg loss.
            loss_plan_bound (Tensor): planning map boundary constraint loss.
            loss_plan_col (Tensor): planning col constraint loss.
            loss_plan_dir (Tensor): planning directional constraint loss.
        """
        if path_loss:
            self.mode = self.pw_ego_fut_mode
            use_col_loss = False # Path的时候不用
        else:
            self.mode = self.ego_fut_mode

            if self.use_col_loss:
                use_col_loss = True
            else:
                use_col_loss = False
        ego_fut_gt = ego_fut_gt.unsqueeze(1).repeat(1, self.mode , 1, 1)
        loss_plan_l1_weight = ego_fut_cmd[..., None, None] * ego_fut_masks[:, None, :, None] # ego_fut_masks (4,6) ego_fut_cmd (4,6)
        loss_plan_l1_weight = loss_plan_l1_weight.repeat(1, 1, 1, 2) # 4 6 6 1

        loss_plan_l1 = self.loss_plan_reg(
            ego_fut_preds, # (4, 6, 6, 2)
            ego_fut_gt,
            loss_plan_l1_weight
        )

        if lane_preds is not None and lane_score_preds is not None:
            loss_plan_bound = self.loss_plan_bound(
                ego_fut_preds[ego_fut_cmd==1],
                lane_preds,
                lane_score_preds,
                weight=ego_fut_masks,
                denormalize=False,
            )
        if use_col_loss:
            loss_plan_col = self.loss_plan_col(
                ego_fut_preds[ego_fut_cmd==1],
                agent_preds,
                agent_fut_preds,
                agent_score_preds,
                agent_fut_cls_preds,
                weight=ego_fut_masks[:, :, None].repeat(1, 1, 2)
            )

        # loss_plan_dir = self.loss_plan_dir(
        #     ego_fut_preds[ego_fut_cmd==1],
        #     lane_preds,
        #     lane_score_preds,
        #     weight=ego_fut_masks
        # )

        # if digit_version(TORCH_VERSION) >= digit_version('1.8'):
        #     loss_plan_l1 = torch.nan_to_num(loss_plan_l1)
        #     loss_plan_bound = torch.nan_to_num(loss_plan_bound)
        #     loss_plan_col = torch.nan_to_num(loss_plan_col)
        #     loss_plan_dir = torch.nan_to_num(loss_plan_dir)

        loss_plan_dict = dict()
        loss_plan_dict['loss_plan_reg'] = torch.nan_to_num(loss_plan_l1)
        if lane_preds is not None and lane_score_preds is not None:
            loss_plan_dict['loss_plan_bound'] = torch.nan_to_num(loss_plan_bound)
        if use_col_loss:
            loss_plan_dict['loss_plan_col'] = torch.nan_to_num(loss_plan_col)
            
        # loss_plan_dict['loss_plan_col'] = loss_plan_col
        # loss_plan_dict['loss_plan_dir'] = loss_plan_dir

        return loss_plan_dict


    def loss_planning_diffusion(self,
                      ego_fut_preds,
                      ego_fut_cls,
                      ego_fut_gt,
                      plan_anchor,
                      ego_fut_masks,
                      lane_preds = None,
                      lane_score_preds = None,
                    #   agent_preds,
                    #   agent_fut_preds,
                    #   agent_score_preds,
                    #   agent_fut_cls_preds
                      ):
        bs, num_mode, ts, d = ego_fut_preds.shape
        target_traj = ego_fut_gt
        dist = torch.linalg.norm(target_traj.unsqueeze(1)[...,:2] - plan_anchor, dim=-1)
        dist = dist.mean(dim=-1)
        mode_idx = torch.argmin(dist, dim=-1)
        mode_masks = torch.zeros(*ego_fut_cls.shape[:2],device=ego_fut_cls.device)
        for mask, idx in zip(mode_masks, mode_idx):
            mask[idx] = 1
        mode_masks = mode_masks.to(torch.bool)
        cls_target = mode_idx
        mode_idx = mode_idx[...,None,None,None].repeat(1,1,ts,d)
        best_reg = torch.gather(ego_fut_preds, 1, mode_idx).squeeze(1)
        # import ipdb; ipdb.set_trace()
        # Calculate cls loss using focal loss
        target_classes_onehot = torch.zeros([bs, num_mode],
                                            dtype=ego_fut_cls.dtype,
                                            layout=ego_fut_cls.layout,
                                            device=ego_fut_cls.device)
        target_classes_onehot.scatter_(1, cls_target.unsqueeze(1), 1)

        loss_plan_l1_weight = ego_fut_masks[:, :, None]
        
        loss_plan_l1_weight = loss_plan_l1_weight.repeat(1,  1, 2)

        loss_plan_l1 = self.diff_traj_reg_loss_weight * self.loss_plan_reg(
            best_reg,
            target_traj,
            loss_plan_l1_weight
        )

        if lane_preds is not None and lane_score_preds is not None:
            loss_plan_bound = self.loss_plan_bound(
                best_reg,
                lane_preds,
                lane_score_preds,
                weight=ego_fut_masks,
                denormalize=False,
            )

        
        if self.plan_cls_loss_smooth:
            loss_plan_cls_weight = torch.clip(dist, min=0, max=10.)*10 # scale factor
            loss_plan_cls_weight[mode_masks] = 10.
            loss_plan_cls_weight *= ego_fut_masks.all(dim=-1).to(torch.float).unsqueeze(-1)
            avg_factor = bs*self.ego_fut_mode
            loss_cls = self.diff_traj_cls_loss_weight * py_sigmoid_focal_loss(
                ego_fut_cls,
                target_classes_onehot,
                weight=loss_plan_cls_weight,
                gamma=2.0,
                alpha=0.25,
                reduction='mean',
                avg_factor=avg_factor
            )
        else:
            loss_plan_cls_weight = ego_fut_masks.all(dim=-1).to(torch.float)
            loss_cls = self.diff_traj_cls_loss_weight * py_sigmoid_focal_loss(
                ego_fut_cls,
                target_classes_onehot,
                weight=loss_plan_cls_weight,
                gamma=2.0,
                alpha=0.25,
                reduction='mean',
                avg_factor=None
            )

        return loss_cls, loss_plan_l1, loss_plan_bound

    def plot_latent_distributions(self, present_mu, present_log_sigma, save_path="latent_distributions.png"):
        if not isinstance(present_mu, np.ndarray):
            present_mu = present_mu.detach().cpu().numpy().squeeze() 
        if not isinstance(present_log_sigma, np.ndarray):
            present_log_sigma = present_log_sigma.detach().cpu().numpy().squeeze() 
        
        # 计算标准差σ（假设present_log_sigma是logσ）
        sigma = np.exp(present_log_sigma)
        # 创建子图布局
        fig, axes = plt.subplots(8, 4, figsize=(16, 24))  # 8行4列布局
        axes = axes.ravel()  # 展平为1D数组便于遍历
        
        for i in range(32):
            ax = axes[i]
            mu_i = present_mu[i]
            sigma_i = sigma[i]
            
            # 生成x轴数据（均值±3σ范围）
            x = np.linspace(mu_i - 3*sigma_i, mu_i + 3*sigma_i, 300)
            
            # 计算概率密度函数（PDF）
            pdf = (1/(sigma_i * np.sqrt(2*np.pi))) * np.exp(-0.5*((x - mu_i)/sigma_i)**2)
            
            # 绘制曲线
            ax.plot(x, pdf, color='blue', linewidth=1.5)
            ax.set_title(f'Latent Dimension {i}', fontsize=10)
            ax.set_xlabel('Value', fontsize=8)
            ax.set_ylabel('Density', fontsize=8)
            ax.grid(alpha=0.3)
        
        plt.tight_layout(pad=1.2)
        dir_path = os.path.dirname(save_path)
        if not os.path.exists(dir_path):
            mmcv.mkdir_or_exist(dir_path)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)  # 显式关闭图形释放内存
