_base_ = ["../_base_/datasets/nus-3d.py",
          "../_base_/default_runtime.py"]
backbone_norm_cfg = dict(type='LN', requires_grad=True)

# plugin=True
# plugin_dir='projects/mmdet3d_plugin/'

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]

img_norm_cfg = dict(
   mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# map has classes: divider, ped_crossing, boundary
map_classes = ['Broken','Solid','SolidSolid','Center','TrafficLight','StopSign']

map_fixed_ptsnum_per_gt_line = 11 # now only support fixed_pts > 0
map_eval_use_same_gt_sample_num_flag = True
map_num_classes = len(map_classes)
past_frames = 2
future_frames = 6

_dim_ = 256
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
ida_aug_conf = {
        "resize_lim": (0.37, 0.45),
        "final_dim": (320, 640),
        "bot_pct_lim": (0.0, 0.0),
        "rot_lim": (0.0, 0.0),
        "H": 900,
        "W": 1600,
        "rand_flip": False,
    }

### Occ args ### 
occflow_grid_conf = {
    'xbound': [-50.0, 50.0, 0.5],
    'ybound': [-50.0, 50.0, 0.5],
    'zbound': [-10.0, 10.0, 20.0],
}
# For nuScenes we usually do 10-class detection
NameMapping = {
    #=================vehicle=================
    # bicycle
    'vehicle.bh.crossbike': 'bicycle',
    "vehicle.diamondback.century": 'bicycle',
    "vehicle.gazelle.omafiets": 'bicycle',
    # car
    "vehicle.audi.etron": 'car',
    "vehicle.chevrolet.impala": 'car',
    "vehicle.dodge.charger_2020": 'car',
    "vehicle.dodge.charger_police": 'car',
    "vehicle.dodge.charger_police_2020": 'car',
    "vehicle.lincoln.mkz_2017": 'car',
    "vehicle.lincoln.mkz_2020": 'car',
    "vehicle.mini.cooper_s_2021": 'car',
    "vehicle.mercedes.coupe_2020": 'car',
    "vehicle.ford.mustang": 'car',
    "vehicle.nissan.patrol_2021": 'car',
    "vehicle.audi.tt": 'car',
    "vehicle.audi.etron": 'car',
    "vehicle.ford.crown": 'car',
    "vehicle.ford.mustang": 'car',
    "vehicle.tesla.model3": 'car',
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/FordCrown/SM_FordCrown_parked.SM_FordCrown_parked": 'car',
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Charger/SM_ChargerParked.SM_ChargerParked": 'car',
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Lincoln/SM_LincolnParked.SM_LincolnParked": 'car',
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/MercedesCCC/SM_MercedesCCC_Parked.SM_MercedesCCC_Parked": 'car',
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Mini2021/SM_Mini2021_parked.SM_Mini2021_parked": 'car',
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/NissanPatrol2021/SM_NissanPatrol2021_parked.SM_NissanPatrol2021_parked": 'car',
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/TeslaM3/SM_TeslaM3_parked.SM_TeslaM3_parked": 'car',
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/VolkswagenT2/SM_VolkswagenT2_2021_Parked.SM_VolkswagenT2_2021_Parked": 'car',
    # bus
    # van
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/VolkswagenT2/SM_VolkswagenT2_2021_Parked.SM_VolkswagenT2_2021_Parked": "van",
    "vehicle.ford.ambulance": "van",
    # truck
    "vehicle.carlamotors.firetruck": 'truck',
    #=========================================

    #=================traffic sign============
    # traffic.speed_limit
    "traffic.speed_limit.30": 'traffic_sign',
    "traffic.speed_limit.40": 'traffic_sign',
    "traffic.speed_limit.50": 'traffic_sign',
    "traffic.speed_limit.60": 'traffic_sign',
    "traffic.speed_limit.90": 'traffic_sign',
    "traffic.speed_limit.120": 'traffic_sign',
    
    "traffic.stop": 'traffic_sign',
    "traffic.yield": 'traffic_sign',
    "traffic.traffic_light": 'traffic_light',
    #=========================================

    #===================Construction===========
    "static.prop.warningconstruction" : 'traffic_cone',
    "static.prop.warningaccident": 'traffic_cone',
    "static.prop.trafficwarning": "traffic_cone",

    #===================Construction===========
    "static.prop.constructioncone": 'traffic_cone',

    #=================pedestrian==============
    "walker.pedestrian.0001": 'pedestrian',
    "walker.pedestrian.0003": 'pedestrian',
    "walker.pedestrian.0004": 'pedestrian',
    "walker.pedestrian.0005": 'pedestrian',
    "walker.pedestrian.0007": 'pedestrian',
    "walker.pedestrian.0010": 'pedestrian',
    "walker.pedestrian.0013": 'pedestrian',
    "walker.pedestrian.0014": 'pedestrian',
    "walker.pedestrian.0015": 'pedestrian',
    "walker.pedestrian.0016": 'pedestrian',
    "walker.pedestrian.0017": 'pedestrian',
    "walker.pedestrian.0018": 'pedestrian',
    "walker.pedestrian.0019": 'pedestrian',
    "walker.pedestrian.0020": 'pedestrian',
    "walker.pedestrian.0021": 'pedestrian',
    "walker.pedestrian.0022": 'pedestrian',
    "walker.pedestrian.0025": 'pedestrian',
    "walker.pedestrian.0027": 'pedestrian',
    "walker.pedestrian.0030": 'pedestrian',
    "walker.pedestrian.0031": 'pedestrian',
    "walker.pedestrian.0032": 'pedestrian',
    "walker.pedestrian.0034": 'pedestrian',
    "walker.pedestrian.0035": 'pedestrian',
    "walker.pedestrian.0041": 'pedestrian',
    "walker.pedestrian.0042": 'pedestrian',
    "walker.pedestrian.0046": 'pedestrian',
    "walker.pedestrian.0047": 'pedestrian',

    # ==========================================
    "static.prop.dirtdebris01": 'others',
    "static.prop.dirtdebris02": 'others',
}

class_names = [
'car','van','truck','bicycle','traffic_sign','traffic_cone','traffic_light','pedestrian','others'
]

eval_cfg = {
            "dist_ths": [0.5, 1.0, 2.0, 4.0],
            "dist_th_tp": 2.0,
            "min_recall": 0.1,
            "min_precision": 0.1,
            "mean_ap_weight": 5,
            "class_names":['car','van','truck','bicycle','traffic_sign','traffic_cone','traffic_light','pedestrian'],
            "tp_metrics":['trans_err', 'scale_err', 'orient_err', 'vel_err'],
            "err_name_maping":{'trans_err': 'mATE','scale_err': 'mASE','orient_err': 'mAOE','vel_err': 'mAVE','attr_err': 'mAAE'},
            "class_range":{'car':(50,50),'van':(50,50),'truck':(50,50),'bicycle':(40,40),'traffic_sign':(30,30),'traffic_cone':(30,30),'traffic_light':(30,30),'pedestrian':(40,40)}
            }

queue_length = 1  # each sequence contains `queue_length` frames.
### traj prediction args ###
predict_steps = 12
predict_modes = 6
fut_steps = 4
past_steps = 4
use_nonlinear_optimizer = True
use_memory = True
num_gpus = 32
batch_size = 32
num_iters_per_epoch = 234769 // (num_gpus * batch_size)
num_epochs = 10
llm_path = './Bench2DriveZoo/ckpts/llava-qwen2-0.5b'

use_gen_token = True
collect_keys = ['lidar2img', 'cam_intrinsic', 'timestamp', 'ego_pose', 'ego_pose_inv', 'command']
pretrain = False
is_decoupling = True
use_col_loss = True
use_meta_action = True
rl_training = False
fp32_infer=True
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)

model = dict(
    type='Minddrive',
    save_path='./results_planning_only/',  #save path for vlm models.
    use_grid_mask=True,
    frozen=False,
    use_lora=True,
    open_loop_infer= True,
    is_decoupling = is_decoupling,
    full_train = False,
    lm_model_type = 'qwen2',
    fp32_infer=fp32_infer,
    tokenizer=llm_path,
    lm_head=llm_path, # set to None if don't use llm head
    use_gen_token = use_gen_token,
    rl_training = rl_training, # RL 训练打开
    use_col_loss = use_col_loss,
    use_meta_action = use_meta_action,
    loss_plan_reg=dict(type='L1Loss', loss_weight=3.0),
    loss_plan_bound=dict(type='PlanMapBoundLoss', loss_weight=3.0, dis_thresh=1.0),
    loss_plan_col=dict(type='PlanCollisionLoss', loss_weight=1.0),
    loss_vae_gen=dict(type='ProbabilisticLoss', loss_weight=3.0),
    img_backbone=dict(
        type='EVAViT',
        img_size=640, 
        patch_size=16,
        window_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4*2/3,
        window_block_indexes = (
        list(range(0, 2)) + list(range(3, 5)) + list(range(6, 8)) + list(range(9, 11)) + list(range(12, 14)) + list(range(15, 17)) + list(range(18, 20)) + list(range(21, 23))
        ),
        qkv_bias=True,
        drop_path_rate=0.3,
        flash_attn=True,
        with_cp=True, 
        frozen=False,), 
    map_head=dict(
        type='MinddriveHeadM',
        num_classes=6,
        in_channels=1024,
        out_dims=896,
        memory_len=600,
        with_mask=True, # map query can't see vlm tokens
        topk_proposals=300,
        num_lane=1800,   # 300+1500
        num_lanes_one2one=300,
        k_one2many=5,
        lambda_one2many=1.0,
        num_extra=256,
        n_control=11,
        pc_range=point_cloud_range,
        code_weights = [1.0, 1.0],
        score_threshold=0.2,
        transformer=dict(
            type='PETRTemporalTransformer',
                 input_dimension=256,
                 output_dimension=256,
                 num_layers=6,
                 embed_dims=256,
                 num_heads=8,
                 feedforward_dims=2048,
                 dropout=0.1,
                 with_cp=True,
                 flash_attn=True,),
        train_cfg=dict(
                assigner=dict(
                    type='LaneHungarianAssigner',
                    cls_cost=dict(type='FocalLossCost', weight=1.5),
                    reg_cost=dict(type='LaneL1Cost', weight=0.02),
                    iou_cost=dict(type='IoUCost', weight=0.0))), # dummy
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.5),
        loss_bbox=dict(type='L1Loss', loss_weight=0.02),
        loss_dir=dict(type='PtsDirCosLoss', loss_weight=0.0)), #
    pts_bbox_head=dict(
        type='MinddriveHead',
        num_classes=9,
        in_channels=1024,
        out_dims=896,
        num_query=600,
        with_mask=True,
        memory_len=600,
        topk_proposals=300,
        num_propagated=300,
        num_extra=256,
        n_control=11, # align with centerline query defination
        match_with_velo=False,
        pred_traffic_light_state=True,
        use_memory = use_memory,
        use_col_loss = use_col_loss,
        memory_decoder_transformer = dict(
            type='DREAMTransformerDecoder',
            num_layers=1,
            embed_dims=_dim_,
            num_heads=8,
            dropout=0.0,
            feedforward_dims=_ffn_dim_,
            with_cp=True,
            flash_attn=True,
            return_intermediate=False),
        motion_transformer_decoder=dict(
            type='DREAMTransformerDecoder',
            num_layers=1,
            embed_dims=_dim_,
            num_heads=8,
            dropout=0.0,
            feedforward_dims=_ffn_dim_,
            with_cp=True,
            flash_attn=True,
            return_intermediate=False,
            ),
        motion_map_decoder=dict(
            type='DREAMTransformerDecoder',
            num_layers=1,
            embed_dims=_dim_,
            num_heads=8,
            dropout=0.0,
            feedforward_dims=_ffn_dim_,
            with_cp=True,
            flash_attn=True,
            return_intermediate=False,
            ),
        scalar=10, ##noise groups
        noise_scale = 1.0, 
        dn_weight= 1.0, ##dn loss weight
        split = 0.75, ###positive rate
        code_weights = [2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        score_threshold=0.2,
        class_agnostic_nms=dict(
            classes=[0, 1, 2, 3, 4, 5, 6, 7, 8], 
            compensate=[0, 0, 0.3, 0, 0, 0, 0, 0.3, 0],
            pre_max_size=1000,
            post_max_size=300,
            nms_thr=0.1,
        ),
        transformer=dict(
            type='PETRTemporalTransformer',
                 input_dimension=256,
                 output_dimension=256,
                 num_layers=6,
                 embed_dims=256,
                 num_heads=8,
                 feedforward_dims=2048,
                 dropout=0.1,
                 with_cp=True,
                 flash_attn=True,
            ),
        bbox_coder=dict(
            type='CustomNMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],# 检测到的边界框的中心点的范围。
            pc_range=point_cloud_range, # 
            max_num=300,
            voxel_size=voxel_size,
            num_classes=9), 
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_traffic=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0),),
        # model training and testing settings
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0), # Fake cost. This is just to make it compatible with DETR head. 
            pc_range=point_cloud_range),)
            )
            )


# dataset_type = 'CustomNuScenesDataset'
# data_root = './data/nuscenes/'
# file_client_args = dict(backend='disk')

dataset_type = "B2D_minddrive_Dataset"
dataset_type_rollout = "RL_minddrive_Dataset"

data_root = "data/bench2drive"
info_root = "data/infos"
map_root = "data/bench2drive/maps"
map_file = "data/infos/b2d_map_infos.pkl"

file_client_args = dict(backend="disk")
ann_file_train=info_root + f"/b2d_infos_train.pkl"
ann_file_val=info_root + f"/b2d_infos_val.pkl"
ann_file_test=info_root + f"/b2d_infos_val.pkl"

rollout_buffer = "./carla/rollout_data/eval_bench2drive220_orion_collect_failed_0917_ab_collsion_traffic_light_devi_inter_stop_rollout1/dataset_index.pkl"

train_pipeline = [
    dict(type="LoadMultiViewImageFromFilesInCeph", to_float32=True),
    dict(type="PhotoMetricDistortionMultiViewImage"), 
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=True,with_light_state=True),
    dict(type='VADObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='VADObjectNameFilter', classes=class_names),

    dict(type='LoadAnnoatationHistoryVQA', 
        base_desc_path='./data/critical_desc_v3_cleaned_v1/',
         tokenizer=llm_path, 
         max_length=2048, 
         ignore_type=["v1", "v2", "v3"], # v1 已知2D问3D v2 问3D周围的物体 v3 问lanes上有没有物体 
         use_gen_token=use_gen_token,
         planning_qa_only=True,
         planning_qa_last =True,
         use_meta_action= use_meta_action, 
         is_decoupling=is_decoupling,
         conv_template = "llava_qwen2",
         ),

    # dict(type='RandomScaleImageMultiViewImage', scales=[0.8]), 保持640
    # dict(type='ResizeMultiview3D', img_scale=(640, 640), keep_ratio=False, multiscale_mode='value'),
    dict(type='ResizeCropFlipRotImage', data_aug_conf = ida_aug_conf, training=True),
    dict(type='ResizeMultiview3D', img_scale=(640, 640), keep_ratio=False, multiscale_mode='value'),
    dict(type="PadMultiViewImage", size_divisor=32),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="PETRFormatBundle3D", class_names=class_names, collect_keys = collect_keys),
    dict(type='CustomCollect3D',\
         keys=['gt_bboxes_3d', 'gt_labels_3d', 'img', 'ego_his_trajs','input_ids','gt_attr_labels', 'ego_fut_trajs', 'ego_fut_masks','ego_fut_cmd', 'ego_lcf_feat','vlm_labels','can_bus', 'traffic_state_mask', 'traffic_state','vlm_attn_mask','path_points_future','path_future_mask','cmd_speed','cmd_path']+ collect_keys),
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFilesInCeph', to_float32=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=True),
        dict(type='VADObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='VADObjectNameFilter', classes=class_names),
    
    # dict(type='ResizeMultiview3D', img_scale=(640, 640), keep_ratio=False, multiscale_mode='value'),
    dict(type='ResizeCropFlipRotImage', data_aug_conf = ida_aug_conf, training=False),
    dict(type='ResizeMultiview3D', img_scale=(640, 640), keep_ratio=False, multiscale_mode='value'),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="PadMultiViewImage", size_divisor=32),

    dict(type='LoadAnnoatationMixCriticalVQATest', 
         load_type=["planning"], # please don't test all the questions in single test, it requires quite long time
         tokenizer=llm_path, 
         desc_qa=False,
         use_gen_token=use_gen_token,
         conv_template="llava_qwen2",
         is_decoupling=is_decoupling,
         use_meta_action=use_meta_action,
         single=True,
         max_length=2048,),

    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='PETRFormatBundle3D',
                collect_keys=collect_keys,
                class_names=class_names,
                with_label=False),
            dict(type='CustomCollect3D',\
                keys=['gt_bboxes_3d', 'gt_labels_3d', 'img', 'ego_his_trajs','input_ids','gt_attr_labels', 'ego_fut_trajs', 'ego_fut_masks','ego_fut_cmd', 'ego_lcf_feat','vlm_labels','can_bus','fut_valid_flag','path_points_future','path_future_mask']+collect_keys,
                )]
    )
]

inference_only_pipeline = [
    # dict(type='LoadMultiViewImageFromFilesInCeph', to_float32=True),
    dict(type='ResizeCropFlipRotImage', data_aug_conf = ida_aug_conf, training=False),
    dict(type='ResizeMultiview3D', img_scale=(640, 640), keep_ratio=False, multiscale_mode='value'),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="PadMultiViewImage", size_divisor=32),
    dict(type='LoadAnnoatationMixCriticalVQATest', 
         load_type=["planning"], # please don't test all the questions in single test, it requires quite long time
         tokenizer=llm_path, 
         desc_qa=False,
         use_gen_token=use_gen_token,
         conv_template="llava_qwen2",
         is_decoupling=is_decoupling,
         use_meta_action=use_meta_action,
         single=True,
         max_length=2048,),

    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='PETRFormatBundle3D',
                collect_keys=collect_keys,
                class_names=class_names,
                with_label=False),
            dict(type='CustomCollect3D',\
                keys=['img','input_ids','ego_fut_cmd', 'vlm_labels','can_bus']+collect_keys,
                )]
    ),
]

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=4,
train=dict(
        type=dataset_type,
        seq_mode=True,
        seq_split_num=1,
        data_root=data_root,
        ann_file=ann_file_train,
        pipeline=train_pipeline,
        classes=class_names,
        name_mapping=NameMapping,
        map_root=map_root,
        map_file=map_file,
        modality=input_modality,
        queue_length=queue_length,
        past_frames=past_frames,
        future_frames=future_frames,
        point_cloud_range=point_cloud_range,
        polyline_points_num=map_fixed_ptsnum_per_gt_line, # 每条线的点的个数
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR',
        #custom_eval_version='vad_nusc_detection_cvpr_2019'
        ),
    val=dict(type=dataset_type,
            data_root=data_root,
            ann_file=ann_file_val,
            pipeline=test_pipeline,
            classes=class_names,
            name_mapping=NameMapping,
            map_root=map_root,
            map_file=map_file,
            modality=input_modality,
            queue_length=queue_length,
            past_frames=past_frames,
            future_frames=future_frames,
            point_cloud_range=point_cloud_range,
            polyline_points_num=map_fixed_ptsnum_per_gt_line,
            eval_cfg=eval_cfg
            ),
    test=dict(type=dataset_type,
            data_root=data_root,
            ann_file=ann_file_val,
            pipeline=test_pipeline,
            classes=class_names,
            name_mapping=NameMapping,
            map_root=map_root,
            map_file=map_file,
            modality=input_modality,
            queue_length=queue_length,
            past_frames=past_frames,
            future_frames=future_frames,
            point_cloud_range=point_cloud_range,
            polyline_points_num=map_fixed_ptsnum_per_gt_line,
            eval_cfg=eval_cfg
            ),
    shuffle =True,
    shuffler_sampler=dict(
        type="InfiniteGroupEachSampleInBatchSampler",
        seq_split_num=10,
        warmup_split_num=80, # lane det and vlm need short term temporal fusion in the early stage of training
        num_iters_to_seq=num_iters_per_epoch,),
    nonshuffler_sampler=dict(type="DistributedSampler"),
    )


optimizer = dict(constructor='LearningRateDecayOptimizerConstructor', type='AdamW', 
                 lr=8e-5, betas=(0.9, 0.999), weight_decay=1e-5,
                 paramwise_cfg={'decay_rate': 0.9,
                                'head_decay_rate': 4.0,
                                'lm_head_decay_rate': 0.1,
                                'decay_type': 'vit_wise',
                                'num_layers': 24,
                                })

optimizer_config = dict(type='Fp16OptimizerHook', loss_scale='dynamic', grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
    )

# evaluation = dict(interval=num_iters_per_epoch*(num_epochs+1), pipeline=test_pipeline) # no eval
find_unused_parameters=False #### when use checkpoint, find_unused_parameters must be False
no_use_kl_and_entro = True
workflow = [('ppo_train', num_epochs * num_iters_per_epoch)]
# checkpoint_config = dict(interval=100, max_keep_ckpts=3) # check
checkpoint_config = dict(interval=num_iters_per_epoch, max_keep_ckpts=3)
runner = dict(
    type='RLIterBasedRunner', max_iters=num_epochs * num_iters_per_epoch, il_step=20, n_steps=6000)

log_config = dict(
    interval=10, hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")]
)
load_from='work_dirs/orion_hisv2_tl_mm_ml_qwenv2_pretrain_lora_mixqa_stage3_meta_action_decouple_long/iter_11004_v4_minddrive.pth'
resume_from=None