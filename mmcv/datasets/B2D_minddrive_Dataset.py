# ------------------------------------------------------------------------
# Modified from Bench2Drive(https://github.com/Thinklab-SJTU/Bench2Drive)
# Copyright (c) Xiaomi Corporation. All rights reserved.
# ------------------------------------------------------------------------
import copy
import numpy as np
import os
from os import path as osp
import torch
import random
import json, pickle
import tempfile
import cv2
import pyquaternion
from pyquaternion import Quaternion
import mmcv
from mmcv.datasets import DATASETS
from mmcv.utils import save_tensor
from mmcv.parallel import DataContainer as DC
from mmcv.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes
from mmcv.fileio.io import load, dump
from mmcv.utils import track_iter_progress, mkdir_or_exist
from mmcv.datasets.pipelines import to_tensor
from .custom_3d import Custom3DDataset
from .pipelines import Compose
from mmcv.datasets.map_utils.struct import LiDARInstanceLines
from shapely.geometry import LineString
import random
from nuscenes.utils.data_classes import Box as NuScenesBox
from mmcv.core.bbox.structures.nuscenes_box import CustomNuscenesBox
from shapely import affinity, ops
from shapely.geometry import LineString, box, MultiPolygon, MultiLineString
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from nuscenes.eval.detection.constants import DETECTION_NAMES
from mmcv.datasets.map_utils.mean_ap import eval_map
from mmcv.datasets.map_utils.mean_ap import format_res_gt_by_classes
from .nuscenes_styled_eval_utils import DetectionMetrics, EvalBoxes, DetectionBox,center_distance,accumulate,DetectionMetricDataList,calc_ap, calc_tp, quaternion_yaw
import math

@DATASETS.register_module()
class B2D_minddrive_Dataset(Custom3DDataset):
    def __init__(self, queue_length=4, bev_size=(200, 200), seq_mode=False, seq_split_num=1, overlap_test=False,with_velocity=True,sample_interval=5, space_num = 11,name_mapping= None,eval_cfg = None, map_root =None,map_file=None,past_frames=2, future_frames=6,point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0] ,polyline_points_num=20,*args, eval_mode=['lane', 'det'], **kwargs):
        super().__init__(*args, **kwargs)
        self.queue_length = queue_length
        self.bev_size = bev_size
        self.overlap_test = overlap_test
        self.with_velocity = with_velocity
        self.NameMapping  = name_mapping
        self.eval_cfg  = eval_cfg
        self.sample_interval = sample_interval
        self.past_frames = past_frames
        self.future_frames = future_frames
        self.map_root = map_root
        self.map_file = map_file
        self.space_num = space_num
        self.point_cloud_range = np.array(point_cloud_range)
        self.polyline_points_num = polyline_points_num
        self.map_element_class = {'Broken':0, 'Solid':1, 'SolidSolid':2,'Center':3,'TrafficLight':4,'StopSign':5}
        self.MAPCLASSES = list(self.map_element_class.keys())
        self.NUM_MAPCLASSES = len(self.MAPCLASSES)
        self.map_eval_use_same_gt_sample_num_flag = True
        self.map_ann_file = 'data/infos'
        self.eval_cfg  = eval_cfg
        self.eval_mode = eval_mode
        with open(self.map_file,'rb') as f: 
            self.map_infos = pickle.load(f)
        if seq_mode:
            self.num_frame_losses = 1
            self.queue_length = 1
            self.seq_split_num = seq_split_num
            self.random_length = 0
            self._set_sequence_group_flag() # Must be called after load_annotations b/c load_annotations does sorting.

    def _set_sequence_group_flag(self):
        """
        Set each sequence to be a different group
        """
        res = []

        curr_sequence = 0
        curr_scene_token = self.data_infos[0]['folder']

        for idx in range(len(self.data_infos)):
            if idx != 0 and self.data_infos[idx]['folder'] != curr_scene_token:
                # Not first frame and # new folder -> new sequence
                curr_sequence += 1
                curr_scene_token = self.data_infos[idx]['folder']
            res.append(curr_sequence)

        self.flag = np.array(res, dtype=np.int64)

        if self.seq_split_num != 1:
            if self.seq_split_num == 'all':
                self.flag = np.array(range(len(self.data_infos)), dtype=np.int64)
            else:
                bin_counts = np.bincount(self.flag)
                new_flags = []
                curr_new_flag = 0
                for curr_flag in range(len(bin_counts)):
                    curr_sequence_length = np.array(
                        list(range(0, 
                                bin_counts[curr_flag], 
                                math.ceil(bin_counts[curr_flag] / self.seq_split_num)))
                        + [bin_counts[curr_flag]])

                    for sub_seq_idx in (curr_sequence_length[1:] - curr_sequence_length[:-1]):
                        for _ in range(sub_seq_idx):
                            new_flags.append(curr_new_flag)
                        curr_new_flag += 1

                assert len(new_flags) == len(self.flag)
                assert len(np.bincount(new_flags)) == len(np.bincount(self.flag)) * self.seq_split_num
                self.flag = np.array(new_flags, dtype=np.int64)

    def invert_pose(self, pose):
        inv_pose = np.eye(4)
        inv_pose[:3, :3] = np.transpose(pose[:3, :3])
        inv_pose[:3, -1] = - inv_pose[:3, :3] @ pose[:3, -1]
        return inv_pose

    def prepare_train_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        queue = []
        index_list = list(range(index-self.queue_length*self.sample_interval, index,self.sample_interval))
        random.shuffle(index_list)
        index_list = sorted(index_list[1:])
        index_list.append(index)
        for i in index_list:
            i = max(0, i)
            input_dict = self.get_data_info(i)
            if input_dict is None:
                return None
            self.pre_pipeline(input_dict)
            example = self.pipeline(input_dict)
            gt_labels,gt_bboxes = self.get_map_info(index)
            example['map_gt_labels_3d'] = DC(gt_labels, cpu_only=False)
            example['map_gt_bboxes_3d'] = DC(gt_bboxes, cpu_only=True)
            
            if self.filter_empty_gt and \
                    (example is None or ~(example['gt_labels_3d']._data != -1).any()):
                return None
            queue.append(example)
        return self.union2one(queue)


    def union2one(self, queue):
        imgs_list = [each['img'].data for each in queue]
        metas_map = {}
        prev_scene_token = None
        prev_pos = None
        prev_angle = None
        for i, each in enumerate(queue):
            metas_map[i] = each['img_metas'].data
        queue[-1]['img'] = DC(torch.stack(imgs_list), cpu_only=False, stack=True)
        queue[-1]['img_metas'] = DC(metas_map, cpu_only=True)
        queue = queue[-1]
        return queue

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]

        for i in range(len(info['gt_names'])):
            if info['gt_names'][i] in self.NameMapping.keys():
                info['gt_names'][i] = self.NameMapping[info['gt_names'][i]]
        
        ego2global = np.linalg.inv(info['sensors']['LIDAR_TOP']['world2lidar'])  # in fact, lidar2global
        ego_pose = ego2global
        
        ego_pose_inv = invert_matrix_egopose_numpy(ego_pose)
        
        input_dict = dict(
            folder=info['folder'],
            scene_token=info['folder'],
            frame_idx=info['frame_idx'],
            ego_yaw=np.nan_to_num(info['ego_yaw'],nan=np.pi/2),
            ego_translation=info['ego_translation'],
            sensors=info['sensors'],
            ego_pose=ego_pose,
            ego_pose_inv = ego_pose_inv,
            world2lidar=info['sensors']['LIDAR_TOP']['world2lidar'],
            lidar2ego=info['sensors']['LIDAR_TOP']['lidar2ego'],
            gt_ids=info['gt_ids'],
            gt_boxes=info['gt_boxes'],
            gt_names=info['gt_names'],
            ego_vel = info['ego_vel'],
            ego_accel = info['ego_accel'],
            ego_rotation_rate = info['ego_rotation_rate'],
            npc2world = info['npc2world'],
            timestamp=info['frame_idx']/10
        )

        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            lidar2ego = info['sensors']['LIDAR_TOP']['lidar2ego']
            lidar2global =  self.invert_pose(info['sensors']['LIDAR_TOP']['world2lidar'])
            for sensor_type, cam_info in info['sensors'].items():
                if not 'CAM' in sensor_type:
                    continue
                image_paths.append(osp.join(self.data_root,cam_info['data_path']))
                # obtain lidar to image transformation matrix
                cam2ego = cam_info['cam2ego']
                intrinsic = cam_info['intrinsic']
                intrinsic_pad = np.eye(4)
                intrinsic_pad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2cam = self.invert_pose(cam2ego) @ lidar2ego
                lidar2img = intrinsic_pad @ lidar2cam
                lidar2img_rts.append(lidar2img)
                cam_intrinsics.append(intrinsic_pad)
                lidar2cam_rts.append(lidar2cam)

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsics,
                    lidar2cam = lidar2cam_rts,
                    l2g_r_mat=lidar2global[0:3,0:3],
                    l2g_t=lidar2global[0:3,3]

                ))
            
        #if not self.test_mode:
        annos = self.get_ann_info(index)
        input_dict['ann_info'] = annos
        yaw = input_dict['ego_yaw']
        rotation = list(Quaternion(axis=[0, 0, 1], radians=yaw))
        
        if yaw < 0:
            yaw += 2*np.pi
        yaw_in_degree = yaw / np.pi * 180 
        
        can_bus = np.zeros(18)
        can_bus[:3] = input_dict['ego_translation']
        can_bus[3:7] = rotation
        can_bus[7:10] = input_dict['ego_vel']
        can_bus[10:13] = input_dict['ego_accel']
        can_bus[13:16] = input_dict['ego_rotation_rate']
        can_bus[16] = yaw
        can_bus[17] = yaw_in_degree
        input_dict['can_bus'] = can_bus
        ego_lcf_feat = np.zeros(9)
        ego_lcf_feat[0:2] = input_dict['ego_translation'][0:2]
        ego_lcf_feat[2:4] = input_dict['ego_accel'][2:4]
        ego_lcf_feat[4] = input_dict['ego_rotation_rate'][-1]
        ego_lcf_feat[5] = info['ego_size'][1]
        ego_lcf_feat[6] = info['ego_size'][0]
        ego_lcf_feat[7] = np.sqrt(input_dict['ego_translation'][0]**2+input_dict['ego_translation'][1]**2)
        ego_lcf_feat[8] = info['steer']
        ego_his_trajs, ego_fut_trajs, ego_fut_masks, command, command_nohot = self.get_ego_trajs(index,self.sample_interval,self.past_frames,self.future_frames)
        path_points_future, path_future_mask = self.get_ego_path_traj(index,space_num=21)
        input_dict['ego_his_trajs'] = ego_his_trajs
        input_dict['ego_fut_trajs'] = ego_fut_trajs
        input_dict['path_points_future'] = path_points_future
        input_dict['path_future_mask'] = path_future_mask
        input_dict['ego_fut_masks'] = ego_fut_masks
        input_dict['ego_fut_cmd'] = command
        input_dict['command'] = command_nohot
        input_dict['ego_lcf_feat'] = ego_lcf_feat
        input_dict['fut_valid_flag'] = (ego_fut_masks==1).all() 

        return input_dict


    def get_map_info(self, index):

        gt_masks = []
        gt_labels = []
        gt_bboxes = []

        ann_info = self.data_infos[index]
        town_name = ann_info['town_name']
        map_info = self.map_infos[town_name]
        lane_points = map_info['lane_points']
        lane_sample_points = map_info['lane_sample_points']
        lane_types = map_info['lane_types']
        trigger_volumes_points = map_info['trigger_volumes_points']
        trigger_volumes_sample_points = map_info['trigger_volumes_sample_points']
        trigger_volumes_types = map_info['trigger_volumes_types']
        world2lidar = np.array(ann_info['sensors']['LIDAR_TOP']['world2lidar'])
        ego_xy = np.linalg.inv(world2lidar)[0:2,3]
        max_distance = 50
        chosed_idx = []

        for idx in range(len(lane_sample_points)):
            single_sample_points = lane_sample_points[idx]
            distance = np.linalg.norm((single_sample_points[:,0:2]-ego_xy),axis=-1)
            if np.min(distance) < max_distance:
                chosed_idx.append(idx) # 选出自车周围50m的车道线

        polylines = []

        for idx in chosed_idx:
            if not lane_types[idx] in self.map_element_class.keys(): # Broken 0  Solid 1 SolidSolid 2 Center 3 TrafficLight 4  StopSign 5                         
                continue
            points = lane_points[idx]
            points = np.concatenate([points,np.ones((points.shape[0],1))],axis=-1)
            points_in_lidar = (world2lidar @ points.T).T
            mask = (points_in_lidar[:,0]>self.point_cloud_range[0]) & (points_in_lidar[:,0]<self.point_cloud_range[3]) & (points_in_lidar[:,1]>self.point_cloud_range[1]) & (points_in_lidar[:,1]<self.point_cloud_range[4])
            points_in_lidar_range = points_in_lidar[mask,0:2]
            if len(points_in_lidar_range) > 1:
                polylines.append(LineString(points_in_lidar_range))
                gt_label =  self.map_element_class[lane_types[idx]]
                gt_labels.append(gt_label)

        for idx in range(len(trigger_volumes_points)):
            if not trigger_volumes_types[idx] in self.map_element_class.keys():
                continue
            points = trigger_volumes_points[idx]
            points = np.concatenate([points,np.ones((points.shape[0],1))],axis=-1)
            points_in_lidar = (world2lidar @ points.T).T
            mask = (points_in_lidar[:,0]>self.point_cloud_range[0]) & (points_in_lidar[:,0]<self.point_cloud_range[3]) & (points_in_lidar[:,1]>self.point_cloud_range[1]) & (points_in_lidar[:,1]<self.point_cloud_range[4])
            points_in_lidar_range = points_in_lidar[mask,0:2]
            if mask.all():
                polylines.append(LineString(np.concatenate((points_in_lidar_range,points_in_lidar_range[0:1]),axis=0)))
                gt_label = self.map_element_class[trigger_volumes_types[idx]]
                gt_labels.append(gt_label)

        gt_labels = torch.tensor(gt_labels)
        gt_bboxes = LiDARInstanceLines(polylines,fixed_num=self.polyline_points_num,patch_size=(self.point_cloud_range[4]-self.point_cloud_range[1],self.point_cloud_range[3]-self.point_cloud_range[0]))
        return gt_labels, gt_bboxes

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): \
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        info = self.data_infos[index]
        # filter out bbox containing no points

        for i in range(len(info['gt_names'])):
            if info['gt_names'][i] in self.NameMapping.keys():
                info['gt_names'][i] = self.NameMapping[info['gt_names'][i]]
        mask = (info['num_points'] != 0)
        gt_bboxes_3d = info['gt_boxes'][mask]
        gt_names_3d = info['gt_names'][mask]
        gt_inds = info['gt_ids'][mask]
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        if not self.with_velocity:
            gt_bboxes_3d = gt_bboxes_3d[:,0:7]

        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)
        attr_labels = self.get_box_attr_labels(index,self.sample_interval,self.future_frames)
        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
            gt_ids=gt_inds,
            attr_labels=attr_labels[mask],
            )
        return anns_results

    def __getitem__(self, idx):
        """Get item from infos according to the given index.
        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:

            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data


    def get_ego_path_traj(self, idx, space_num):
        current_frame = self.data_infos[idx]
        world2lidar_current = current_frame['sensors']['LIDAR_TOP']['world2lidar']
        
        trajectory = [np.zeros(2)]
        accumulated_dist = 0.0
        current_idx = idx
        accumulated_dist_list =[]
        accumulated_dist_list.append(accumulated_dist)
        
        while True:
            current_idx += 1
            if current_idx >= len(self.data_infos):
                break
            next_frame = self.data_infos[current_idx]
            if next_frame['folder'] != current_frame['folder']:
                break

            world2lidar_next = next_frame['sensors']['LIDAR_TOP']['world2lidar']
            next2current = world2lidar_current @ np.linalg.inv(world2lidar_next)
            new_point = next2current[0:2, 3]

            delta = np.linalg.norm(new_point - trajectory[-1])
            trajectory.append(new_point)
            accumulated_dist += delta
            accumulated_dist_list.append(accumulated_dist)
            
            if accumulated_dist >= 30.0:
                break
            
        path_mask = np.ones(space_num-1) 
        target_distances = np.arange(0, space_num, 1.0)
        sampled_trajectory = []

        for s in target_distances:
            if accumulated_dist_list[-1] < s:
                sampled_trajectory.append(trajectory[-1])
                continue
            
            idx = np.searchsorted(accumulated_dist_list, s, side='right') - 1
            if idx < 0:
                idx = 0
            if idx >= len(trajectory) - 1:
                sampled_trajectory.append(trajectory[-1])
                continue
            
            segment_length = accumulated_dist_list[idx+1] - accumulated_dist_list[idx]
            if segment_length < 1e-6:
                sampled_trajectory.append(trajectory[idx])
            else:
                t = (s - accumulated_dist_list[idx]) / segment_length
                interpolated_point = trajectory[idx] + t * (trajectory[idx+1] - trajectory[idx])
                sampled_trajectory.append(interpolated_point)

        sampled_trajectory = np.array(sampled_trajectory)
        sampled_trajectory = sampled_trajectory[1:] - sampled_trajectory[:-1]
        return sampled_trajectory, path_mask


    def get_ego_trajs(self,idx,sample_rate,past_frames,future_frames):

        adj_idx_list = range(idx-past_frames*sample_rate,idx+(future_frames+1)*sample_rate,sample_rate)
        cur_frame = self.data_infos[idx]
        full_adj_track = np.zeros((past_frames+future_frames+1,2))
        full_adj_adj_mask = np.zeros(past_frames+future_frames+1)
        world2lidar_lidar_cur = cur_frame['sensors']['LIDAR_TOP']['world2lidar']
        for j in range(len(adj_idx_list)):
            adj_idx = adj_idx_list[j]
            if adj_idx <0 or adj_idx>=len(self.data_infos):
                break
            adj_frame = self.data_infos[adj_idx]
            if adj_frame['folder'] != cur_frame ['folder']:
                break
            world2lidar_ego_adj = adj_frame['sensors']['LIDAR_TOP']['world2lidar']
            adj2cur_lidar = world2lidar_lidar_cur @ np.linalg.inv(world2lidar_ego_adj)
            xy = adj2cur_lidar[0:2,3]
            full_adj_track[j,0:2] = xy
            full_adj_adj_mask[j] = 1
        offset_track = full_adj_track[1:] - full_adj_track[:-1]
        for j in range(past_frames-1,-1,-1):
            if full_adj_adj_mask[j] == 0:
                offset_track[j] = offset_track[j+1]
        for j in range(past_frames,past_frames+future_frames,1):

            if full_adj_adj_mask[j+1] == 0 :
                offset_track[j] = 0
        command = self.command2hot(cur_frame['command_near'])
        command_nohot = self.command2nohot(cur_frame['command_near'])
        return offset_track[:past_frames].copy(), offset_track[past_frames:].copy(), full_adj_adj_mask[-future_frames:].copy(), command, command_nohot
    
    def command2hot(self,command,max_dim=6):
        if command < 0:
            command = 4
        command -= 1
        cmd_one_hot = np.zeros(max_dim)
        cmd_one_hot[command] = 1
        return cmd_one_hot

    def command2nohot(self,command,max_dim=6):
        if command < 0:
            command = 4
        command -= 1
        return command
    
    def get_box_attr_labels(self,idx,sample_rate,frames):


        adj_idx_list = range(idx,idx+(frames+1)*sample_rate,sample_rate)
        cur_frame = self.data_infos[idx]
        cur_box_names = cur_frame['gt_names']
        for i in range(len(cur_box_names)):
            if cur_box_names[i] in self.NameMapping.keys():
                cur_box_names[i] = self.NameMapping[cur_box_names[i]]
        cur_boxes = cur_frame['gt_boxes'].copy()
        box_ids = cur_frame['gt_ids']
        future_track = np.zeros((len(box_ids),frames+1,2))
        future_mask = np.zeros((len(box_ids),frames+1))
        future_yaw = np.zeros((len(box_ids),frames+1))
        gt_fut_goal = np.zeros((len(box_ids),1))
        agent_lcf_feat = np.zeros((len(box_ids),9))
        world2lidar_lidar_cur = cur_frame['sensors']['LIDAR_TOP']['world2lidar']
        for i in range(len(box_ids)):
            agent_lcf_feat[i,0:2] = cur_boxes[i,0:2]
            agent_lcf_feat[i,2] = cur_boxes[i,6]
            agent_lcf_feat[i,3:5] = cur_boxes[i,7:]
            agent_lcf_feat[i,5:8] = cur_boxes[i,3:6]
            cur_box_name = cur_box_names[i]
            if cur_box_name in self.CLASSES:
                agent_lcf_feat[i, 8] = self.CLASSES.index(cur_box_name)
            else:
                agent_lcf_feat[i, 8] = -1

            box_id = box_ids[i]
            cur_box2lidar = world2lidar_lidar_cur @ cur_frame['npc2world'][i]
            cur_xy = cur_box2lidar[0:2,3]      
            for j in range(len(adj_idx_list)):
                adj_idx = adj_idx_list[j]
                if adj_idx <0 or adj_idx>=len(self.data_infos):
                    break
                adj_frame = self.data_infos[adj_idx]
                if adj_frame['folder'] != cur_frame ['folder']:
                    break
                if len(np.where(adj_frame['gt_ids']==box_id)[0])==0:
                    continue
                assert len(np.where(adj_frame['gt_ids']==box_id)[0]) == 1 , np.where(adj_frame['gt_ids']==box_id)[0]
                adj_idx = np.where(adj_frame['gt_ids']==box_id)[0][0]
                adj_box2lidar = world2lidar_lidar_cur @ adj_frame['npc2world'][adj_idx]
                adj_xy = adj_box2lidar[0:2,3]    
                future_track[i,j,:] = adj_xy
                future_mask[i,j] = 1
                future_yaw[i,j] = np.arctan2(adj_box2lidar[1,0],adj_box2lidar[0,0])

            coord_diff = future_track[i,-1] - future_track[i,0]
            if coord_diff.max() < 1.0: # static
                gt_fut_goal[i] = 9
            else:
                box_mot_yaw = np.arctan2(coord_diff[1], coord_diff[0]) + np.pi
                gt_fut_goal[i] = box_mot_yaw // (np.pi / 4)  # 0-8: goal direction class

        future_track_offset = future_track[:,1:,:] - future_track[:,:-1,:]
        future_mask_offset = future_mask[:,1:]
        future_track_offset[future_mask_offset==0] = 0
        future_yaw_offset = future_yaw[:,1:] - future_yaw[:,:-1]
        mask1 = np.where(future_yaw_offset>np.pi)
        mask2 = np.where(future_yaw_offset<-np.pi)
        future_yaw_offset[mask1] -=np.pi*2 
        future_yaw_offset[mask2] +=np.pi*2
        attr_labels = np.concatenate([future_track_offset.reshape(-1,frames*2), future_mask_offset, gt_fut_goal, agent_lcf_feat, future_yaw_offset],axis=-1).astype(np.float32)
        return attr_labels.copy()



    def load_gt(self):
        all_annotations = EvalBoxes()
        for i in range(len(self.data_infos)):
            sample_boxes = []
            sample_data = self.data_infos[i]
            gt_boxes = sample_data['gt_boxes']
            for j in range(gt_boxes.shape[0]):
                class_name = self.NameMapping[sample_data['gt_names'][j]]
                if not class_name in self.eval_cfg['class_range'].keys():
                    continue
                range_x, range_y = self.eval_cfg['class_range'][class_name]
                if abs(gt_boxes[j,0]) > range_x or abs(gt_boxes[j,1]) > range_y:
                    continue
                sample_boxes.append(DetectionBox(
                                                sample_token=sample_data['folder']+'_'+str(sample_data['frame_idx']),
                                                translation=gt_boxes[j,0:3],
                                                size=gt_boxes[j,3:6],
                                                rotation=list(Quaternion(axis=[0, 0, 1], radians=-gt_boxes[j,6]-np.pi/2)),
                                                velocity=gt_boxes[j,7:9],
                                                num_pts=int(sample_data['num_points'][j]),
                                                detection_name=class_name,
                                                detection_score=-1.0,  
                                                attribute_name=class_name
                                                ))
            all_annotations.add_boxes(sample_data['folder']+'_'+str(sample_data['frame_idx']), sample_boxes)
        return all_annotations



    def _format_gt(self):
        gt_annos = []
        print('Start to convert gt map format...')
        # assert self.map_ann_file is not None
        if (not os.path.exists(self.map_ann_file)) :
            dataset_length = len(self)
            prog_bar = mmcv.ProgressBar(dataset_length)
            mapped_class_names = self.MAPCLASSES
            for sample_id in range(dataset_length):
                sample_token = self.data_infos[sample_id]['folder'] +  '_' + str(self.data_infos[sample_id]['frame_idx'])
                gt_anno = {}
                gt_anno['sample_token'] = sample_token
                # gt_sample_annos = []
                gt_sample_dict = {}
                gt_labels , gt_bboxes = self.get_map_info(sample_id)
                gt_vecs = gt_bboxes.instance_list
                gt_vec_list = []
                for i, (gt_label, gt_vec) in enumerate(zip(gt_labels, gt_vecs)):
                    name = mapped_class_names[gt_label]
                    anno = dict(
                        pts=np.array(list(gt_vec.coords)),
                        pts_num=len(list(gt_vec.coords)),
                        cls_name=name,
                        type=gt_label,
                    )
                    gt_vec_list.append(anno)
                gt_anno['vectors']=gt_vec_list
                gt_annos.append(gt_anno)

                prog_bar.update()
            nusc_submissions = {
                'GTs': gt_annos
            }
            print('\n GT anns writes to', self.map_ann_file)
            dump(nusc_submissions, self.map_ann_file)
        else:
            print(f'{self.map_ann_file} exist, not update')


    def _format_bbox(self, results, jsonfile_prefix=None, score_thresh=0.2):
        """Convert the results to the standard format.

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of the output jsonfile.
                You can specify the output directory/filename by
                modifying the jsonfile_prefix. Default: None.

        Returns:
            str: Path of the output json file.
        """

        nusc_annos = {}
        det_mapped_class_names = self.CLASSES
        # assert self.map_ann_file is not None
        map_pred_annos = {}
        map_mapped_class_names = self.MAPCLASSES
        plan_annos = {}
        print('Start to convert detection format...')
        for sample_id, det in enumerate(track_iter_progress(results)):
            #pdb.set_trace()
            annos = []
            box3d = det['boxes_3d']
            scores = det['scores_3d']
            labels = det['labels_3d']
            box_gravity_center = box3d.gravity_center
            box_dims = box3d.dims
            box_yaw = box3d.yaw.numpy()
            box_yaw = -box_yaw - np.pi / 2
            sample_token = self.data_infos[sample_id]['folder'] + '_' + str(self.data_infos[sample_id]['frame_idx'])
            for i in range(len(box3d)):
                #import pdb;pdb.set_trace()
                if scores[i] < score_thresh:
                    continue
                quat = list(Quaternion(axis=[0, 0, 1], radians=box_yaw[i]))
                velocity = [box3d.tensor[i, 7].item(),box3d.tensor[i, 8].item()]
                name = det_mapped_class_names[labels[i]]
                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box_gravity_center[i].tolist(),
                    size=box_dims[i].tolist(),
                    rotation=quat,
                    velocity=velocity,
                    detection_name=name,
                    detection_score=scores[i].item(),
                    attribute_name=name)
                annos.append(nusc_anno)
            nusc_annos[sample_token] = annos
            map_pred_anno = {}
            vecs = output_to_vecs(det)
            sample_token = self.data_infos[sample_id]['folder'] +  '_' + str(self.data_infos[sample_id]['frame_idx'])
            map_pred_anno['sample_token'] = sample_token
            pred_vec_list=[]
            for i, vec in enumerate(vecs):
                name = map_mapped_class_names[vec['label']]
                anno = dict(
                    # sample_token=sample_token,
                    pts=vec['pts'],
                    pts_num=len(vec['pts']),
                    cls_name=name,
                    type=vec['label'],
                    confidence_level=vec['score'])
                pred_vec_list.append(anno)
            map_pred_anno['vectors'] = pred_vec_list
            map_pred_annos[sample_token] = map_pred_anno


        nusc_submissions = {
            'meta': self.modality,
            'results': nusc_annos,
            'map_results': map_pred_annos,
            'plan_results': plan_annos
            # 'GTs': gt_annos
        }

        mmcv.mkdir_or_exist(jsonfile_prefix)

        res_path = osp.join(jsonfile_prefix, 'results_nusc.json')
        print('Results writes to', res_path)
        dump(nusc_submissions, res_path)
        return res_path

    def format_results(self, results, jsonfile_prefix=None):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: Returns (result_files, tmp_dir), where `result_files` is a \
                dict containing the json filepaths, `tmp_dir` is the temporal \
                directory created for saving json files when \
                `jsonfile_prefix` is not specified.
        """
        if isinstance(results, dict):
            # print(f'results must be a list, but get dict, keys={results.keys()}')
            # assert isinstance(results, list)
            results = results['bbox_results']
        assert isinstance(results, list)
        # assert len(results) == len(self), (
        #     'The length of results is not equal to the dataset len: {} != {}'.
        #     format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        # currently the output prediction results could be in two formats
        # 1. list of dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...)
        # 2. list of dict('pts_bbox' or 'img_bbox':
        #     dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...))
        # this is a workaround to enable evaluation of both formats on nuScenes
        # refer to https://github.com/open-mmlab/mmdetection3d/issues/449
        if not ('pts_bbox' in results[0] or 'img_bbox' in results[0]):
            result_files = self._format_bbox(results, jsonfile_prefix)
        else:
            # should take the inner dict out of 'pts_bbox' or 'img_bbox' dict
            result_files = dict()
            for name in results[0]:
                if name == 'metric_results':
                    continue
                if name == 'text_out':
                    continue
                print(f'\nFormating bboxes of {name}')
                results_ = [out[name] for out in results]
                tmp_file_ = osp.join(jsonfile_prefix, name)
                result_files.update(
                    {name: self._format_bbox(results_, tmp_file_)})
        return result_files, tmp_dir

    def _evaluate_single(self,
                         result_path,
                         logger=None,
                         metric='bbox',
                         map_metric='chamfer',
                         result_name='pts_bbox'):
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            metric (str): Metric name used for evaluation. Default: 'bbox'.
            result_name (str): Result name in the metric prefix.
                Default: 'pts_bbox'.

        Returns:
            dict: Dictionary of evaluation details.
        """
        detail = dict()
        with open(result_path,'r') as f:
            result_data = json.load(f)
        pred_boxes = EvalBoxes.deserialize(result_data['results'], DetectionBox)
        meta = result_data['meta']



        gt_boxes = self.load_gt()

        metric_data_list = DetectionMetricDataList()
        for class_name in self.eval_cfg['class_names']:
            for dist_th in self.eval_cfg['dist_ths']:
                md = accumulate(gt_boxes, pred_boxes, class_name, center_distance, dist_th)
                metric_data_list.set(class_name, dist_th, md)
                metrics = DetectionMetrics(self.eval_cfg)

        for class_name in self.eval_cfg['class_names']:
            # Compute APs.
            for dist_th in self.eval_cfg['dist_ths']:
                metric_data = metric_data_list[(class_name, dist_th)]
                ap = calc_ap(metric_data, self.eval_cfg['min_recall'], self.eval_cfg['min_precision'])
                metrics.add_label_ap(class_name, dist_th, ap)

            # Compute TP metrics.
            for metric_name in self.eval_cfg['tp_metrics']:
                metric_data = metric_data_list[(class_name, self.eval_cfg['dist_th_tp'])]
                tp = calc_tp(metric_data, self.eval_cfg['min_recall'], metric_name)
                metrics.add_label_tp(class_name, metric_name, tp)

        metrics_summary = metrics.serialize()
        metrics_summary['meta'] = meta.copy()
        print('mAP: %.4f' % (metrics_summary['mean_ap']))
        err_name_mapping = {
            'trans_err': 'mATE',
            'scale_err': 'mASE',
            'orient_err': 'mAOE',
            'vel_err': 'mAVE',
        }
        for tp_name, tp_val in metrics_summary['tp_errors'].items():
            print('%s: %.4f' % (err_name_mapping[tp_name], tp_val))
        print('NDS: %.4f' % (metrics_summary['nd_score']))
        print()
        print('Per-class results:')
        print('Object Class\tAP\tATE\tASE\tAOE\tAVE')
        class_aps = metrics_summary['mean_dist_aps']
        class_tps = metrics_summary['label_tp_errors']
        for class_name in class_aps.keys():
            print('%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
                  % (class_name, class_aps[class_name],
                     class_tps[class_name]['trans_err'],
                     class_tps[class_name]['scale_err'],
                     class_tps[class_name]['orient_err'],
                     class_tps[class_name]['vel_err']))        

        detail = dict()
        metric_prefix = 'bbox_NuScenes'
        for name in self.eval_cfg['class_names']:
            for k, v in metrics_summary['label_aps'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_AP_dist_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics_summary['label_tp_errors'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics_summary['tp_errors'].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}'.format(metric_prefix,self.eval_cfg['err_name_maping'][k])] = val
        detail['{}/NDS'.format(metric_prefix)] = metrics_summary['nd_score']
        detail['{}/mAP'.format(metric_prefix)] = metrics_summary['mean_ap']

        return detail
    
    def evaluate(self,
                 results,
                 metric='bbox',
                 map_metric='chamfer',
                 logger=None,
                 jsonfile_prefix=None,
                 result_names=['pts_bbox'],
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluation in nuScenes protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """
        
        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)
        # NOTE: print planning metric
        print('\n')
        print('-------------- Planning --------------')
        metric_dict = None
        num_valid = 0
        for res in results:
            if res['metric_results']['fut_valid_flag']:
                num_valid += 1
            else:
                continue
            if metric_dict is None:
                metric_dict = copy.deepcopy(res['metric_results'])
            else:
                for k in res['metric_results'].keys():
                    metric_dict[k] += res['metric_results'][k]
        
        for k in metric_dict:
            metric_dict[k] = metric_dict[k] / num_valid
            print("{}:{}".format(k, metric_dict[k]))

        if isinstance(result_files, dict):
            results_dict = dict()
            for name in result_names:
                print('Evaluating bboxes of {}'.format(name))
                ret_dict = self._evaluate_single(result_files[name], metric=metric, map_metric=map_metric)
            results_dict.update(ret_dict)
        elif isinstance(result_files, str):
            results_dict = self._evaluate_single(result_files, metric=metric, map_metric=map_metric)

        if tmp_dir is not None:
            tmp_dir.cleanup()

        if show:
            self.show(results, out_dir, pipeline=pipeline)
        return results_dict

def output_to_nusc_box(detection):
    """Convert the output to the box class in the nuScenes.

    Args:
        detection (dict): Detection results.

            - boxes_3d (:obj:`BaseInstance3DBoxes`): Detection bbox.
            - scores_3d (torch.Tensor): Detection scores.
            - labels_3d (torch.Tensor): Predicted box labels.

    Returns:
        list[:obj:`NuScenesBox`]: List of standard NuScenesBoxes.
    """
    box3d = detection['boxes_3d']
    scores = detection['scores_3d'].numpy()
    labels = detection['labels_3d'].numpy()
    trajs = detection['trajs_3d'].numpy()


    box_gravity_center = box3d.gravity_center.numpy()
    box_dims = box3d.dims.numpy()
    box_yaw = box3d.yaw.numpy()
    # TODO: check whether this is necessary
    # with dir_offset & dir_limit in the head
    box_yaw = -box_yaw - np.pi / 2

    box_list = []
    for i in range(len(box3d)):
        quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
        velocity = (*box3d.tensor[i, 7:9], 0.0)
        # velo_val = np.linalg.norm(box3d[i, 7:9])
        # velo_ori = box3d[i, 6]
        # velocity = (
        # velo_val * np.cos(velo_ori), velo_val * np.sin(velo_ori), 0.0)
        box = CustomNuscenesBox(
            center=box_gravity_center[i],
            size=box_dims[i],
            orientation=quat,
            fut_trajs=trajs[i],
            label=labels[i],
            score=scores[i],
            velocity=velocity)
        box_list.append(box)
    return box_list


def lidar_nusc_box_to_global(info,
                             boxes,
                             classes,
                             eval_configs,
                             eval_version='detection_cvpr_2019'):
    """Convert the box from ego to global coordinate.

    Args:
        info (dict): Info for a specific sample data, including the
            calibration information.
        boxes (list[:obj:`NuScenesBox`]): List of predicted NuScenesBoxes.
        classes (list[str]): Mapped classes in the evaluation.
        eval_configs (object): Evaluation configuration object.
        eval_version (str): Evaluation version.
            Default: 'detection_cvpr_2019'

    Returns:
        list: List of standard NuScenesBoxes in the global
            coordinate.
    """
    box_list = []
    for box in boxes:
        # Move box to ego vehicle coord system
        box.rotate(pyquaternion.Quaternion(info['lidar2ego_rotation']))
        box.translate(np.array(info['lidar2ego_translation']))
        # filter det in ego.
        cls_range_x_map = eval_configs.class_range_x
        cls_range_y_map = eval_configs.class_range_y
        x_distance, y_distance = box.center[0], box.center[1]
        det_range_x = cls_range_x_map[classes[box.label]]
        det_range_y = cls_range_y_map[classes[box.label]]
        if abs(x_distance) > det_range_x or abs(y_distance) > det_range_y:
            continue
        # Move box to global coord system
        box.rotate(pyquaternion.Quaternion(info['ego2global_rotation']))
        box.translate(np.array(info['ego2global_translation']))
        box_list.append(box)
    return box_list

def output_to_vecs(detection):
    # box3d = detection['map_boxes_3d'].numpy()
    scores = detection['map_scores_3d'].numpy()
    labels = detection['map_labels_3d'].numpy()
    pts = detection['map_pts_3d'].numpy()

    vec_list = []
    # import pdb;pdb.set_trace()
    for i in range(labels.shape[0]):
        vec = dict(
            # bbox = box3d[i], # xyxy
            label=labels[i],
            score=scores[i],
            pts=pts[i],
        )
        vec_list.append(vec)
    return vec_list

def invert_matrix_egopose_numpy(egopose):
    """ Compute the inverse transformation of a 4x4 egopose numpy matrix."""
    inverse_matrix = np.zeros((4, 4), dtype=np.float32)
    rotation = egopose[:3, :3]
    translation = egopose[:3, 3]
    inverse_matrix[:3, :3] = rotation.T
    inverse_matrix[:3, 3] = -np.dot(rotation.T, translation)
    inverse_matrix[3, 3] = 1.0
    return inverse_matrix

def convert_egopose_to_matrix_numpy(rotation, translation):
    transformation_matrix = np.zeros((4, 4), dtype=np.float32)
    transformation_matrix[:3, :3] = rotation
    transformation_matrix[:3, 3] = translation
    transformation_matrix[3, 3] = 1.0
    return transformation_matrix