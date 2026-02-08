# Modified from Orion(https://github.com/xiaomi-mlab/Orion)
# Copyright (c) Xiaomi Corporation. All rights reserved.
# ------------------------------------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
from unittest import result
import numpy as np
from numpy import random
import warnings
from mmcv import is_tuple_of
from mmcv.utils import build_from_cfg
from mmcv.parallel import DataContainer as DC

from mmcv.core.voxel.voxel_generator import VoxelGenerator
from mmcv.core.bbox.structures.cam_box3d import CameraInstance3DBoxes
from mmcv.core.bbox.structures.depth_box3d import DepthInstance3DBoxes
from mmcv.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes
from mmcv.core.bbox import box_np_ops
from mmcv.datasets.builder import PIPELINES
from mmcv.image import impad, impad_to_multiple, imnormalize, imresize,is_list_of ,bgr2hsv, hsv2bgr, imrescale
from ..builder import OBJECTSAMPLERS

from transformers import AutoTokenizer
import json
import re
import os
import copy
import torch
import math
import pickle
from shapely.geometry import MultiPoint, Polygon, LineString, Point
from shapely.geometry import box as canvas_box
from nuscenes.utils.geometry_utils import view_points

from ..data_utils.constants import DEFAULT_IMAGE_TOKEN
from ..data_utils.data_utils import preprocess, preprocess_llama3, preprocess_qwen2
from ..data_utils import conversation as conversation_lib
from PIL import Image
from pathlib import Path


@PIPELINES.register_module()
class RandomDropPointsColor(object):
    r"""Randomly set the color of points to all zeros.

    Once this transform is executed, all the points' color will be dropped.
    Refer to `PAConv <https://github.com/CVMI-Lab/PAConv/blob/main/scene_seg/
    util/transform.py#L223>`_ for more details.

    Args:
        drop_ratio (float): The probability of dropping point colors.
            Defaults to 0.2.
    """

    def __init__(self, drop_ratio=0.2):
        assert isinstance(drop_ratio, (int, float)) and 0 <= drop_ratio <= 1, \
            f'invalid drop_ratio value {drop_ratio}'
        self.drop_ratio = drop_ratio

    def __call__(self, input_dict):
        """Call function to drop point colors.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after color dropping, \
                'points' key is updated in the result dict.
        """
        points = input_dict['points']
        assert points.attribute_dims is not None and \
            'color' in points.attribute_dims, \
            'Expect points have color attribute'

        # this if-expression is a bit strange
        # `RandomDropPointsColor` is used in training 3D segmentor PAConv
        # we discovered in our experiments that, using
        # `if np.random.rand() > 1.0 - self.drop_ratio` consistently leads to
        # better results than using `if np.random.rand() < self.drop_ratio`
        # so we keep this hack in our codebase
        if np.random.rand() > 1.0 - self.drop_ratio:
            points.color = points.color * 0.0
        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(drop_ratio={self.drop_ratio})'
        return repr_str


@PIPELINES.register_module()
class RandomJitterPoints(object):
    """Randomly jitter point coordinates.

    Different from the global translation in ``GlobalRotScaleTrans``, here we \
        apply different noises to each point in a scene.

    Args:
        jitter_std (list[float]): The standard deviation of jittering noise.
            This applies random noise to all points in a 3D scene, which is \
            sampled from a gaussian distribution whose standard deviation is \
            set by ``jitter_std``. Defaults to [0.01, 0.01, 0.01]
        clip_range (list[float] | None): Clip the randomly generated jitter \
            noise into this range. If None is given, don't perform clipping.
            Defaults to [-0.05, 0.05]

    Note:
        This transform should only be used in point cloud segmentation tasks \
            because we don't transform ground-truth bboxes accordingly.
        For similar transform in detection task, please refer to `ObjectNoise`.
    """

    def __init__(self,
                 jitter_std=[0.01, 0.01, 0.01],
                 clip_range=[-0.05, 0.05]):
        seq_types = (list, tuple, np.ndarray)
        if not isinstance(jitter_std, seq_types):
            assert isinstance(jitter_std, (int, float)), \
                f'unsupported jitter_std type {type(jitter_std)}'
            jitter_std = [jitter_std, jitter_std, jitter_std]
        self.jitter_std = jitter_std

        if clip_range is not None:
            if not isinstance(clip_range, seq_types):
                assert isinstance(clip_range, (int, float)), \
                    f'unsupported clip_range type {type(clip_range)}'
                clip_range = [-clip_range, clip_range]
        self.clip_range = clip_range

    def __call__(self, input_dict):
        """Call function to jitter all the points in the scene.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after adding noise to each point, \
                'points' key is updated in the result dict.
        """
        points = input_dict['points']
        jitter_std = np.array(self.jitter_std, dtype=np.float32)
        jitter_noise = \
            np.random.randn(points.shape[0], 3) * jitter_std[None, :]
        if self.clip_range is not None:
            jitter_noise = np.clip(jitter_noise, self.clip_range[0],
                                   self.clip_range[1])

        points.translate(jitter_noise)
        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(jitter_std={self.jitter_std},'
        repr_str += f' clip_range={self.clip_range})'
        return repr_str


@PIPELINES.register_module()
class ObjectSample(object):
    """Sample GT objects to the data.

    Args:
        db_sampler (dict): Config dict of the database sampler.
        sample_2d (bool): Whether to also paste 2D image patch to the images
            This should be true when applying multi-modality cut-and-paste.
            Defaults to False.
    """

    def __init__(self, db_sampler, sample_2d=False):
        self.sampler_cfg = db_sampler
        self.sample_2d = sample_2d
        if 'type' not in db_sampler.keys():
            db_sampler['type'] = 'DataBaseSampler'
        self.db_sampler = build_from_cfg(db_sampler, OBJECTSAMPLERS)

    @staticmethod
    def remove_points_in_boxes(points, boxes):
        """Remove the points in the sampled bounding boxes.

        Args:
            points (:obj:`BasePoints`): Input point cloud array.
            boxes (np.ndarray): Sampled ground truth boxes.

        Returns:
            np.ndarray: Points with those in the boxes removed.
        """
        masks = box_np_ops.points_in_rbbox(points.coord.numpy(), boxes)
        points = points[np.logical_not(masks.any(-1))]
        return points

    def __call__(self, input_dict):
        """Call function to sample ground truth objects to the data.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after object sampling augmentation, \
                'points', 'gt_bboxes_3d', 'gt_labels_3d' keys are updated \
                in the result dict.
        """
        gt_bboxes_3d = input_dict['gt_bboxes_3d']
        gt_labels_3d = input_dict['gt_labels_3d']

        # change to float for blending operation
        points = input_dict['points']
        if self.sample_2d:
            img = input_dict['img']
            gt_bboxes_2d = input_dict['gt_bboxes']
            # Assume for now 3D & 2D bboxes are the same
            sampled_dict = self.db_sampler.sample_all(
                gt_bboxes_3d.tensor.numpy(),
                gt_labels_3d,
                gt_bboxes_2d=gt_bboxes_2d,
                img=img)
        else:
            sampled_dict = self.db_sampler.sample_all(
                gt_bboxes_3d.tensor.numpy(), gt_labels_3d, img=None)

        if sampled_dict is not None:
            sampled_gt_bboxes_3d = sampled_dict['gt_bboxes_3d']
            sampled_points = sampled_dict['points']
            sampled_gt_labels = sampled_dict['gt_labels_3d']

            gt_labels_3d = np.concatenate([gt_labels_3d, sampled_gt_labels],
                                          axis=0)
            gt_bboxes_3d = gt_bboxes_3d.new_box(
                np.concatenate(
                    [gt_bboxes_3d.tensor.numpy(), sampled_gt_bboxes_3d]))

            points = self.remove_points_in_boxes(points, sampled_gt_bboxes_3d)
            # check the points dimension
            points = points.cat([sampled_points, points])

            if self.sample_2d:
                sampled_gt_bboxes_2d = sampled_dict['gt_bboxes_2d']
                gt_bboxes_2d = np.concatenate(
                    [gt_bboxes_2d, sampled_gt_bboxes_2d]).astype(np.float32)

                input_dict['gt_bboxes'] = gt_bboxes_2d
                input_dict['img'] = sampled_dict['img']

        input_dict['gt_bboxes_3d'] = gt_bboxes_3d
        input_dict['gt_labels_3d'] = gt_labels_3d.astype(np.long)
        input_dict['points'] = points

        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f' sample_2d={self.sample_2d},'
        repr_str += f' data_root={self.sampler_cfg.data_root},'
        repr_str += f' info_path={self.sampler_cfg.info_path},'
        repr_str += f' rate={self.sampler_cfg.rate},'
        repr_str += f' prepare={self.sampler_cfg.prepare},'
        repr_str += f' classes={self.sampler_cfg.classes},'
        repr_str += f' sample_groups={self.sampler_cfg.sample_groups}'
        return repr_str




@PIPELINES.register_module()
class GlobalAlignment(object):
    """Apply global alignment to 3D scene points by rotation and translation.

    Args:
        rotation_axis (int): Rotation axis for points and bboxes rotation.

    Note:
        We do not record the applied rotation and translation as in \
            GlobalRotScaleTrans. Because usually, we do not need to reverse \
            the alignment step.
        For example, ScanNet 3D detection task uses aligned ground-truth \
            bounding boxes for evaluation.
    """

    def __init__(self, rotation_axis):
        self.rotation_axis = rotation_axis

    def _trans_points(self, input_dict, trans_factor):
        """Private function to translate points.

        Args:
            input_dict (dict): Result dict from loading pipeline.
            trans_factor (np.ndarray): Translation vector to be applied.

        Returns:
            dict: Results after translation, 'points' is updated in the dict.
        """
        input_dict['points'].translate(trans_factor)

    def _rot_points(self, input_dict, rot_mat):
        """Private function to rotate bounding boxes and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.
            rot_mat (np.ndarray): Rotation matrix to be applied.

        Returns:
            dict: Results after rotation, 'points' is updated in the dict.
        """
        # input should be rot_mat_T so I transpose it here
        input_dict['points'].rotate(rot_mat.T)

    def _check_rot_mat(self, rot_mat):
        """Check if rotation matrix is valid for self.rotation_axis.

        Args:
            rot_mat (np.ndarray): Rotation matrix to be applied.
        """
        is_valid = np.allclose(np.linalg.det(rot_mat), 1.0)
        valid_array = np.zeros(3)
        valid_array[self.rotation_axis] = 1.0
        is_valid &= (rot_mat[self.rotation_axis, :] == valid_array).all()
        is_valid &= (rot_mat[:, self.rotation_axis] == valid_array).all()
        assert is_valid, f'invalid rotation matrix {rot_mat}'

    def __call__(self, input_dict):
        """Call function to shuffle points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after global alignment, 'points' and keys in \
                input_dict['bbox3d_fields'] are updated in the result dict.
        """
        assert 'axis_align_matrix' in input_dict['ann_info'].keys(), \
            'axis_align_matrix is not provided in GlobalAlignment'

        axis_align_matrix = input_dict['ann_info']['axis_align_matrix']
        assert axis_align_matrix.shape == (4, 4), \
            f'invalid shape {axis_align_matrix.shape} for axis_align_matrix'
        rot_mat = axis_align_matrix[:3, :3]
        trans_vec = axis_align_matrix[:3, -1]

        self._check_rot_mat(rot_mat)
        self._rot_points(input_dict, rot_mat)
        self._trans_points(input_dict, trans_vec)

        return input_dict

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(rotation_axis={self.rotation_axis})'
        return repr_str


@PIPELINES.register_module()
class GlobalRotScaleTrans(object):
    """Apply global rotation, scaling and translation to a 3D scene.

    Args:
        rot_range (list[float]): Range of rotation angle.
            Defaults to [-0.78539816, 0.78539816] (close to [-pi/4, pi/4]).
        scale_ratio_range (list[float]): Range of scale ratio.
            Defaults to [0.95, 1.05].
        translation_std (list[float]): The standard deviation of translation
            noise. This applies random translation to a scene by a noise, which
            is sampled from a gaussian distribution whose standard deviation
            is set by ``translation_std``. Defaults to [0, 0, 0]
        shift_height (bool): Whether to shift height.
            (the fourth dimension of indoor points) when scaling.
            Defaults to False.
    """

    def __init__(self,
                 rot_range=[-0.78539816, 0.78539816],
                 scale_ratio_range=[0.95, 1.05],
                 translation_std=[0, 0, 0],
                 shift_height=False):
        seq_types = (list, tuple, np.ndarray)
        if not isinstance(rot_range, seq_types):
            assert isinstance(rot_range, (int, float)), \
                f'unsupported rot_range type {type(rot_range)}'
            rot_range = [-rot_range, rot_range]
        self.rot_range = rot_range

        assert isinstance(scale_ratio_range, seq_types), \
            f'unsupported scale_ratio_range type {type(scale_ratio_range)}'
        self.scale_ratio_range = scale_ratio_range

        if not isinstance(translation_std, seq_types):
            assert isinstance(translation_std, (int, float)), \
                f'unsupported translation_std type {type(translation_std)}'
            translation_std = [
                translation_std, translation_std, translation_std
            ]
        assert all([std >= 0 for std in translation_std]), \
            'translation_std should be positive'
        self.translation_std = translation_std
        self.shift_height = shift_height

    def _trans_bbox_points(self, input_dict):
        """Private function to translate bounding boxes and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after translation, 'points', 'pcd_trans' \
                and keys in input_dict['bbox3d_fields'] are updated \
                in the result dict.
        """
        translation_std = np.array(self.translation_std, dtype=np.float32)
        trans_factor = np.random.normal(scale=translation_std, size=3).T

        input_dict['points'].translate(trans_factor)
        input_dict['pcd_trans'] = trans_factor
        for key in input_dict['bbox3d_fields']:
            input_dict[key].translate(trans_factor)

    def _rot_bbox_points(self, input_dict):
        """Private function to rotate bounding boxes and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after rotation, 'points', 'pcd_rotation' \
                and keys in input_dict['bbox3d_fields'] are updated \
                in the result dict.
        """
        rotation = self.rot_range
        noise_rotation = np.random.uniform(rotation[0], rotation[1])

        # if no bbox in input_dict, only rotate points
        if len(input_dict['bbox3d_fields']) == 0:
            rot_mat_T = input_dict['points'].rotate(noise_rotation)
            input_dict['pcd_rotation'] = rot_mat_T
            return

        # rotate points with bboxes
        for key in input_dict['bbox3d_fields']:
            if len(input_dict[key].tensor) != 0:
                points, rot_mat_T = input_dict[key].rotate(
                    noise_rotation, input_dict['points'])
                input_dict['points'] = points
                input_dict['pcd_rotation'] = rot_mat_T

    def _scale_bbox_points(self, input_dict):
        """Private function to scale bounding boxes and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'points'and keys in \
                input_dict['bbox3d_fields'] are updated in the result dict.
        """
        scale = input_dict['pcd_scale_factor']
        points = input_dict['points']
        points.scale(scale)
        if self.shift_height:
            assert 'height' in points.attribute_dims.keys(), \
                'setting shift_height=True but points have no height attribute'
            points.tensor[:, points.attribute_dims['height']] *= scale
        input_dict['points'] = points

        for key in input_dict['bbox3d_fields']:
            input_dict[key].scale(scale)

    def _random_scale(self, input_dict):
        """Private function to randomly set the scale factor.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'pcd_scale_factor' are updated \
                in the result dict.
        """
        scale_factor = np.random.uniform(self.scale_ratio_range[0],
                                         self.scale_ratio_range[1])
        input_dict['pcd_scale_factor'] = scale_factor

    def __call__(self, input_dict):
        """Private function to rotate, scale and translate bounding boxes and \
        points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'points', 'pcd_rotation',
                'pcd_scale_factor', 'pcd_trans' and keys in \
                input_dict['bbox3d_fields'] are updated in the result dict.
        """
        if 'transformation_3d_flow' not in input_dict:
            input_dict['transformation_3d_flow'] = []

        self._rot_bbox_points(input_dict)

        if 'pcd_scale_factor' not in input_dict:
            self._random_scale(input_dict)
        self._scale_bbox_points(input_dict)

        self._trans_bbox_points(input_dict)

        input_dict['transformation_3d_flow'].extend(['R', 'S', 'T'])
        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(rot_range={self.rot_range},'
        repr_str += f' scale_ratio_range={self.scale_ratio_range},'
        repr_str += f' translation_std={self.translation_std},'
        repr_str += f' shift_height={self.shift_height})'
        return repr_str


@PIPELINES.register_module()
class PointShuffle(object):
    """Shuffle input points."""

    def __call__(self, input_dict):
        """Call function to shuffle points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'points', 'pts_instance_mask' \
                and 'pts_semantic_mask' keys are updated in the result dict.
        """
        idx = input_dict['points'].shuffle()
        idx = idx.numpy()

        pts_instance_mask = input_dict.get('pts_instance_mask', None)
        pts_semantic_mask = input_dict.get('pts_semantic_mask', None)

        if pts_instance_mask is not None:
            input_dict['pts_instance_mask'] = pts_instance_mask[idx]

        if pts_semantic_mask is not None:
            input_dict['pts_semantic_mask'] = pts_semantic_mask[idx]

        return input_dict

    def __repr__(self):
        return self.__class__.__name__


@PIPELINES.register_module()
class ObjectRangeFilter(object):
    """Filter objects by the range.

    Args:
        point_cloud_range (list[float]): Point cloud range.
    """

    def __init__(self, point_cloud_range):
        self.pcd_range = np.array(point_cloud_range, dtype=np.float32)

    def __call__(self, input_dict):
        """Call function to filter objects by the range.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d' \
                keys are updated in the result dict.
        """
        # Check points instance type and initialise bev_range
        if isinstance(input_dict['gt_bboxes_3d'],
                      (LiDARInstance3DBoxes, DepthInstance3DBoxes)):
            bev_range = self.pcd_range[[0, 1, 3, 4]]
        elif isinstance(input_dict['gt_bboxes_3d'], CameraInstance3DBoxes):
            bev_range = self.pcd_range[[0, 2, 3, 5]]

        gt_bboxes_3d = input_dict['gt_bboxes_3d']
        gt_labels_3d = input_dict['gt_labels_3d']
        mask = gt_bboxes_3d.in_range_bev(bev_range)
        gt_bboxes_3d = gt_bboxes_3d[mask]
        # mask is a torch tensor but gt_labels_3d is still numpy array
        # using mask to index gt_labels_3d will cause bug when
        # len(gt_labels_3d) == 1, where mask=1 will be interpreted
        # as gt_labels_3d[1] and cause out of index error
        gt_labels_3d = gt_labels_3d[mask.numpy().astype(np.bool)]

        # limit rad to [-pi, pi]
        gt_bboxes_3d.limit_yaw(offset=0.5, period=2 * np.pi)
        input_dict['gt_bboxes_3d'] = gt_bboxes_3d
        input_dict['gt_labels_3d'] = gt_labels_3d

        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(point_cloud_range={self.pcd_range.tolist()})'
        return repr_str


@PIPELINES.register_module()
class PointsRangeFilter(object):
    """Filter points by the range.

    Args:
        point_cloud_range (list[float]): Point cloud range.
    """

    def __init__(self, point_cloud_range):
        self.pcd_range = np.array(point_cloud_range, dtype=np.float32)

    def __call__(self, input_dict):
        """Call function to filter points by the range.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'points', 'pts_instance_mask' \
                and 'pts_semantic_mask' keys are updated in the result dict.
        """
        points = input_dict['points']
        points_mask = points.in_range_3d(self.pcd_range)
        clean_points = points[points_mask]
        input_dict['points'] = clean_points
        points_mask = points_mask.numpy()

        pts_instance_mask = input_dict.get('pts_instance_mask', None)
        pts_semantic_mask = input_dict.get('pts_semantic_mask', None)

        if pts_instance_mask is not None:
            input_dict['pts_instance_mask'] = pts_instance_mask[points_mask]

        if pts_semantic_mask is not None:
            input_dict['pts_semantic_mask'] = pts_semantic_mask[points_mask]

        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(point_cloud_range={self.pcd_range.tolist()})'
        return repr_str


@PIPELINES.register_module()
class ObjectNameFilter(object):
    """Filter GT objects by their names.

    Args:
        classes (list[str]): List of class names to be kept for training.
    """

    def __init__(self, classes):
        self.classes = classes
        self.labels = list(range(len(self.classes)))

    def __call__(self, input_dict):
        """Call function to filter objects by their names.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d' \
                keys are updated in the result dict.
        """
        gt_labels_3d = input_dict['gt_labels_3d']
        gt_bboxes_mask = np.array([n in self.labels for n in gt_labels_3d],
                                  dtype=np.bool_)
        input_dict['gt_bboxes_3d'] = input_dict['gt_bboxes_3d'][gt_bboxes_mask]
        input_dict['gt_labels_3d'] = input_dict['gt_labels_3d'][gt_bboxes_mask]

        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(classes={self.classes})'
        return repr_str


@PIPELINES.register_module()
class PointSample(object):
    """Point sample.

    Sampling data to a certain number.

    Args:
        num_points (int): Number of points to be sampled.
        sample_range (float, optional): The range where to sample points.
            If not None, the points with depth larger than `sample_range` are
            prior to be sampled. Defaults to None.
        replace (bool, optional): Whether the sampling is with or without
            replacement. Defaults to False.
    """

    def __init__(self, num_points, sample_range=None, replace=False):
        self.num_points = num_points
        self.sample_range = sample_range
        self.replace = replace

    def _points_random_sampling(self,
                                points,
                                num_samples,
                                sample_range=None,
                                replace=False,
                                return_choices=False):
        """Points random sampling.

        Sample points to a certain number.

        Args:
            points (np.ndarray | :obj:`BasePoints`): 3D Points.
            num_samples (int): Number of samples to be sampled.
            sample_range (float, optional): Indicating the range where the
                points will be sampled. Defaults to None.
            replace (bool, optional): Sampling with or without replacement.
                Defaults to None.
            return_choices (bool, optional): Whether return choice.
                Defaults to False.
        Returns:
            tuple[np.ndarray] | np.ndarray:
                - points (np.ndarray | :obj:`BasePoints`): 3D Points.
                - choices (np.ndarray, optional): The generated random samples.
        """
        if not replace:
            replace = (points.shape[0] < num_samples)
        point_range = range(len(points))
        if sample_range is not None and not replace:
            # Only sampling the near points when len(points) >= num_samples
            depth = np.linalg.norm(points.tensor, axis=1)
            far_inds = np.where(depth > sample_range)[0]
            near_inds = np.where(depth <= sample_range)[0]
            # in case there are too many far points
            if len(far_inds) > num_samples:
                far_inds = np.random.choice(
                    far_inds, num_samples, replace=False)
            point_range = near_inds
            num_samples -= len(far_inds)
        choices = np.random.choice(point_range, num_samples, replace=replace)
        if sample_range is not None and not replace:
            choices = np.concatenate((far_inds, choices))
            # Shuffle points after sampling
            np.random.shuffle(choices)
        if return_choices:
            return points[choices], choices
        else:
            return points[choices]

    def __call__(self, results):
        """Call function to sample points to in indoor scenes.

        Args:
            input_dict (dict): Result dict from loading pipeline.
        Returns:
            dict: Results after sampling, 'points', 'pts_instance_mask' \
                and 'pts_semantic_mask' keys are updated in the result dict.
        """
        points = results['points']
        # Points in Camera coord can provide the depth information.
        # TODO: Need to suport distance-based sampling for other coord system.
        if self.sample_range is not None:
            from mmcv.core.points import CameraPoints
            assert isinstance(points, CameraPoints), \
                'Sampling based on distance is only appliable for CAMERA coord'
        points, choices = self._points_random_sampling(
            points,
            self.num_points,
            self.sample_range,
            self.replace,
            return_choices=True)
        results['points'] = points

        pts_instance_mask = results.get('pts_instance_mask', None)
        pts_semantic_mask = results.get('pts_semantic_mask', None)

        if pts_instance_mask is not None:
            pts_instance_mask = pts_instance_mask[choices]
            results['pts_instance_mask'] = pts_instance_mask

        if pts_semantic_mask is not None:
            pts_semantic_mask = pts_semantic_mask[choices]
            results['pts_semantic_mask'] = pts_semantic_mask

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(num_points={self.num_points},'
        repr_str += f' sample_range={self.sample_range},'
        repr_str += f' replace={self.replace})'

        return repr_str


@PIPELINES.register_module()
class IndoorPointSample(PointSample):
    """Indoor point sample.

    Sampling data to a certain number.
    NOTE: IndoorPointSample is deprecated in favor of PointSample

    Args:
        num_points (int): Number of points to be sampled.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            'IndoorPointSample is deprecated in favor of PointSample')
        super(IndoorPointSample, self).__init__(*args, **kwargs)


@PIPELINES.register_module()
class IndoorPatchPointSample(object):
    r"""Indoor point sample within a patch. Modified from `PointNet++ <https://
    github.com/charlesq34/pointnet2/blob/master/scannet/scannet_dataset.py>`_.

    Sampling data to a certain number for semantic segmentation.

    Args:
        num_points (int): Number of points to be sampled.
        block_size (float, optional): Size of a block to sample points from.
            Defaults to 1.5.
        sample_rate (float, optional): Stride used in sliding patch generation.
            This parameter is unused in `IndoorPatchPointSample` and thus has
            been deprecated. We plan to remove it in the future.
            Defaults to None.
        ignore_index (int, optional): Label index that won't be used for the
            segmentation task. This is set in PointSegClassMapping as neg_cls.
            If not None, will be used as a patch selection criterion.
            Defaults to None.
        use_normalized_coord (bool, optional): Whether to use normalized xyz as
            additional features. Defaults to False.
        num_try (int, optional): Number of times to try if the patch selected
            is invalid. Defaults to 10.
        enlarge_size (float | None, optional): Enlarge the sampled patch to
            [-block_size / 2 - enlarge_size, block_size / 2 + enlarge_size] as
            an augmentation. If None, set it as 0. Defaults to 0.2.
        min_unique_num (int | None, optional): Minimum number of unique points
            the sampled patch should contain. If None, use PointNet++'s method
            to judge uniqueness. Defaults to None.
        eps (float, optional): A value added to patch boundary to guarantee
            points coverage. Defaults to 1e-2.

    Note:
        This transform should only be used in the training process of point
            cloud segmentation tasks. For the sliding patch generation and
            inference process in testing, please refer to the `slide_inference`
            function of `EncoderDecoder3D` class.
    """

    def __init__(self,
                 num_points,
                 block_size=1.5,
                 sample_rate=None,
                 ignore_index=None,
                 use_normalized_coord=False,
                 num_try=10,
                 enlarge_size=0.2,
                 min_unique_num=None,
                 eps=1e-2):
        self.num_points = num_points
        self.block_size = block_size
        self.ignore_index = ignore_index
        self.use_normalized_coord = use_normalized_coord
        self.num_try = num_try
        self.enlarge_size = enlarge_size if enlarge_size is not None else 0.0
        self.min_unique_num = min_unique_num
        self.eps = eps

        if sample_rate is not None:
            warnings.warn(
                "'sample_rate' has been deprecated and will be removed in "
                'the future. Please remove them from your code.')

    def _input_generation(self, coords, patch_center, coord_max, attributes,
                          attribute_dims, point_type):
        """Generating model input.

        Generate input by subtracting patch center and adding additional \
            features. Currently support colors and normalized xyz as features.

        Args:
            coords (np.ndarray): Sampled 3D Points.
            patch_center (np.ndarray): Center coordinate of the selected patch.
            coord_max (np.ndarray): Max coordinate of all 3D Points.
            attributes (np.ndarray): features of input points.
            attribute_dims (dict): Dictionary to indicate the meaning of extra
                dimension.
            point_type (type): class of input points inherited from BasePoints.

        Returns:
            :obj:`BasePoints`: The generated input data.
        """
        # subtract patch center, the z dimension is not centered
        centered_coords = coords.copy()
        centered_coords[:, 0] -= patch_center[0]
        centered_coords[:, 1] -= patch_center[1]

        if self.use_normalized_coord:
            normalized_coord = coords / coord_max
            attributes = np.concatenate([attributes, normalized_coord], axis=1)
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(normalized_coord=[
                    attributes.shape[1], attributes.shape[1] +
                    1, attributes.shape[1] + 2
                ]))

        points = np.concatenate([centered_coords, attributes], axis=1)
        points = point_type(
            points, points_dim=points.shape[1], attribute_dims=attribute_dims)

        return points

    def _patch_points_sampling(self, points, sem_mask):
        """Patch points sampling.

        First sample a valid patch.
        Then sample points within that patch to a certain number.

        Args:
            points (:obj:`BasePoints`): 3D Points.
            sem_mask (np.ndarray): semantic segmentation mask for input points.

        Returns:
            tuple[:obj:`BasePoints`, np.ndarray] | :obj:`BasePoints`:

                - points (:obj:`BasePoints`): 3D Points.
                - choices (np.ndarray): The generated random samples.
        """
        coords = points.coord.numpy()
        attributes = points.tensor[:, 3:].numpy()
        attribute_dims = points.attribute_dims
        point_type = type(points)

        coord_max = np.amax(coords, axis=0)
        coord_min = np.amin(coords, axis=0)

        for _ in range(self.num_try):
            # random sample a point as patch center
            cur_center = coords[np.random.choice(coords.shape[0])]

            # boundary of a patch, which would be enlarged by
            # `self.enlarge_size` as an augmentation
            cur_max = cur_center + np.array(
                [self.block_size / 2.0, self.block_size / 2.0, 0.0])
            cur_min = cur_center - np.array(
                [self.block_size / 2.0, self.block_size / 2.0, 0.0])
            cur_max[2] = coord_max[2]
            cur_min[2] = coord_min[2]
            cur_choice = np.sum(
                (coords >= (cur_min - self.enlarge_size)) *
                (coords <= (cur_max + self.enlarge_size)),
                axis=1) == 3

            if not cur_choice.any():  # no points in this patch
                continue

            cur_coords = coords[cur_choice, :]
            cur_sem_mask = sem_mask[cur_choice]
            point_idxs = np.where(cur_choice)[0]
            mask = np.sum(
                (cur_coords >= (cur_min - self.eps)) * (cur_coords <=
                                                        (cur_max + self.eps)),
                axis=1) == 3

            # two criteria for patch sampling, adopted from PointNet++
            # 1. selected patch should contain enough unique points
            if self.min_unique_num is None:
                # use PointNet++'s method as default
                # [31, 31, 62] are just some big values used to transform
                # coords from 3d array to 1d and then check their uniqueness
                # this is used in all the ScanNet code following PointNet++
                vidx = np.ceil(
                    (cur_coords[mask, :] - cur_min) / (cur_max - cur_min) *
                    np.array([31.0, 31.0, 62.0]))
                vidx = np.unique(vidx[:, 0] * 31.0 * 62.0 + vidx[:, 1] * 62.0 +
                                 vidx[:, 2])
                flag1 = len(vidx) / 31.0 / 31.0 / 62.0 >= 0.02
            else:
                # if `min_unique_num` is provided, directly compare with it
                flag1 = mask.sum() >= self.min_unique_num

            # 2. selected patch should contain enough annotated points
            if self.ignore_index is None:
                flag2 = True
            else:
                flag2 = np.sum(cur_sem_mask != self.ignore_index) / \
                               len(cur_sem_mask) >= 0.7

            if flag1 and flag2:
                break

        # sample idx to `self.num_points`
        if point_idxs.size >= self.num_points:
            # no duplicate in sub-sampling
            choices = np.random.choice(
                point_idxs, self.num_points, replace=False)
        else:
            # do not use random choice here to avoid some points not counted
            dup = np.random.choice(point_idxs.size,
                                   self.num_points - point_idxs.size)
            idx_dup = np.concatenate(
                [np.arange(point_idxs.size),
                 np.array(dup)], 0)
            choices = point_idxs[idx_dup]

        # construct model input
        points = self._input_generation(coords[choices], cur_center, coord_max,
                                        attributes[choices], attribute_dims,
                                        point_type)

        return points, choices

    def __call__(self, results):
        """Call function to sample points to in indoor scenes.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after sampling, 'points', 'pts_instance_mask' \
                and 'pts_semantic_mask' keys are updated in the result dict.
        """
        points = results['points']

        assert 'pts_semantic_mask' in results.keys(), \
            'semantic mask should be provided in training and evaluation'
        pts_semantic_mask = results['pts_semantic_mask']

        points, choices = self._patch_points_sampling(points,
                                                      pts_semantic_mask)

        results['points'] = points
        results['pts_semantic_mask'] = pts_semantic_mask[choices]
        pts_instance_mask = results.get('pts_instance_mask', None)
        if pts_instance_mask is not None:
            results['pts_instance_mask'] = pts_instance_mask[choices]

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(num_points={self.num_points},'
        repr_str += f' block_size={self.block_size},'
        repr_str += f' ignore_index={self.ignore_index},'
        repr_str += f' use_normalized_coord={self.use_normalized_coord},'
        repr_str += f' num_try={self.num_try},'
        repr_str += f' enlarge_size={self.enlarge_size},'
        repr_str += f' min_unique_num={self.min_unique_num},'
        repr_str += f' eps={self.eps})'
        return repr_str


@PIPELINES.register_module()
class BackgroundPointsFilter(object):
    """Filter background points near the bounding box.

    Args:
        bbox_enlarge_range (tuple[float], float): Bbox enlarge range.
    """

    def __init__(self, bbox_enlarge_range):
        assert (is_tuple_of(bbox_enlarge_range, float)
                and len(bbox_enlarge_range) == 3) \
            or isinstance(bbox_enlarge_range, float), \
            f'Invalid arguments bbox_enlarge_range {bbox_enlarge_range}'

        if isinstance(bbox_enlarge_range, float):
            bbox_enlarge_range = [bbox_enlarge_range] * 3
        self.bbox_enlarge_range = np.array(
            bbox_enlarge_range, dtype=np.float32)[np.newaxis, :]

    def __call__(self, input_dict):
        """Call function to filter points by the range.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'points', 'pts_instance_mask' \
                and 'pts_semantic_mask' keys are updated in the result dict.
        """
        points = input_dict['points']
        gt_bboxes_3d = input_dict['gt_bboxes_3d']

        # avoid groundtruth being modified
        gt_bboxes_3d_np = gt_bboxes_3d.tensor.clone().numpy()
        gt_bboxes_3d_np[:, :3] = gt_bboxes_3d.gravity_center.clone().numpy()

        enlarged_gt_bboxes_3d = gt_bboxes_3d_np.copy()
        enlarged_gt_bboxes_3d[:, 3:6] += self.bbox_enlarge_range
        points_numpy = points.tensor.clone().numpy()
        foreground_masks = box_np_ops.points_in_rbbox(
            points_numpy, gt_bboxes_3d_np, origin=(0.5, 0.5, 0.5))
        enlarge_foreground_masks = box_np_ops.points_in_rbbox(
            points_numpy, enlarged_gt_bboxes_3d, origin=(0.5, 0.5, 0.5))
        foreground_masks = foreground_masks.max(1)
        enlarge_foreground_masks = enlarge_foreground_masks.max(1)
        valid_masks = ~np.logical_and(~foreground_masks,
                                      enlarge_foreground_masks)

        input_dict['points'] = points[valid_masks]
        pts_instance_mask = input_dict.get('pts_instance_mask', None)
        if pts_instance_mask is not None:
            input_dict['pts_instance_mask'] = pts_instance_mask[valid_masks]

        pts_semantic_mask = input_dict.get('pts_semantic_mask', None)
        if pts_semantic_mask is not None:
            input_dict['pts_semantic_mask'] = pts_semantic_mask[valid_masks]
        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(bbox_enlarge_range={self.bbox_enlarge_range.tolist()})'
        return repr_str


@PIPELINES.register_module()
class VoxelBasedPointSampler(object):
    """Voxel based point sampler.

    Apply voxel sampling to multiple sweep points.

    Args:
        cur_sweep_cfg (dict): Config for sampling current points.
        prev_sweep_cfg (dict): Config for sampling previous points.
        time_dim (int): Index that indicate the time dimention
            for input points.
    """

    def __init__(self, cur_sweep_cfg, prev_sweep_cfg=None, time_dim=3):
        self.cur_voxel_generator = VoxelGenerator(**cur_sweep_cfg)
        self.cur_voxel_num = self.cur_voxel_generator._max_voxels
        self.time_dim = time_dim
        if prev_sweep_cfg is not None:
            assert prev_sweep_cfg['max_num_points'] == \
                cur_sweep_cfg['max_num_points']
            self.prev_voxel_generator = VoxelGenerator(**prev_sweep_cfg)
            self.prev_voxel_num = self.prev_voxel_generator._max_voxels
        else:
            self.prev_voxel_generator = None
            self.prev_voxel_num = 0

    def _sample_points(self, points, sampler, point_dim):
        """Sample points for each points subset.

        Args:
            points (np.ndarray): Points subset to be sampled.
            sampler (VoxelGenerator): Voxel based sampler for
                each points subset.
            point_dim (int): The dimention of each points

        Returns:
            np.ndarray: Sampled points.
        """
        voxels, coors, num_points_per_voxel = sampler.generate(points)
        if voxels.shape[0] < sampler._max_voxels:
            padding_points = np.zeros([
                sampler._max_voxels - voxels.shape[0], sampler._max_num_points,
                point_dim
            ],
                                      dtype=points.dtype)
            padding_points[:] = voxels[0]
            sample_points = np.concatenate([voxels, padding_points], axis=0)
        else:
            sample_points = voxels

        return sample_points

    def __call__(self, results):
        """Call function to sample points from multiple sweeps.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after sampling, 'points', 'pts_instance_mask' \
                and 'pts_semantic_mask' keys are updated in the result dict.
        """
        points = results['points']
        original_dim = points.shape[1]

        # TODO: process instance and semantic mask while _max_num_points
        # is larger than 1
        # Extend points with seg and mask fields
        map_fields2dim = []
        start_dim = original_dim
        points_numpy = points.tensor.numpy()
        extra_channel = [points_numpy]
        for idx, key in enumerate(results['pts_mask_fields']):
            map_fields2dim.append((key, idx + start_dim))
            extra_channel.append(results[key][..., None])

        start_dim += len(results['pts_mask_fields'])
        for idx, key in enumerate(results['pts_seg_fields']):
            map_fields2dim.append((key, idx + start_dim))
            extra_channel.append(results[key][..., None])

        points_numpy = np.concatenate(extra_channel, axis=-1)

        # Split points into two part, current sweep points and
        # previous sweeps points.
        # TODO: support different sampling methods for next sweeps points
        # and previous sweeps points.
        cur_points_flag = (points_numpy[:, self.time_dim] == 0)
        cur_sweep_points = points_numpy[cur_points_flag]
        prev_sweeps_points = points_numpy[~cur_points_flag]
        if prev_sweeps_points.shape[0] == 0:
            prev_sweeps_points = cur_sweep_points

        # Shuffle points before sampling
        np.random.shuffle(cur_sweep_points)
        np.random.shuffle(prev_sweeps_points)

        cur_sweep_points = self._sample_points(cur_sweep_points,
                                               self.cur_voxel_generator,
                                               points_numpy.shape[1])
        if self.prev_voxel_generator is not None:
            prev_sweeps_points = self._sample_points(prev_sweeps_points,
                                                     self.prev_voxel_generator,
                                                     points_numpy.shape[1])

            points_numpy = np.concatenate(
                [cur_sweep_points, prev_sweeps_points], 0)
        else:
            points_numpy = cur_sweep_points

        if self.cur_voxel_generator._max_num_points == 1:
            points_numpy = points_numpy.squeeze(1)
        results['points'] = points.new_point(points_numpy[..., :original_dim])

        # Restore the correspoinding seg and mask fields
        for key, dim_index in map_fields2dim:
            results[key] = points_numpy[..., dim_index]

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""

        def _auto_indent(repr_str, indent):
            repr_str = repr_str.split('\n')
            repr_str = [' ' * indent + t + '\n' for t in repr_str]
            repr_str = ''.join(repr_str)[:-1]
            return repr_str

        repr_str = self.__class__.__name__
        indent = 4
        repr_str += '(\n'
        repr_str += ' ' * indent + f'num_cur_sweep={self.cur_voxel_num},\n'
        repr_str += ' ' * indent + f'num_prev_sweep={self.prev_voxel_num},\n'
        repr_str += ' ' * indent + f'time_dim={self.time_dim},\n'
        repr_str += ' ' * indent + 'cur_voxel_generator=\n'
        repr_str += f'{_auto_indent(repr(self.cur_voxel_generator), 8)},\n'
        repr_str += ' ' * indent + 'prev_voxel_generator=\n'
        repr_str += f'{_auto_indent(repr(self.prev_voxel_generator), 8)})'
        return repr_str

@PIPELINES.register_module()
class PadMultiViewImage(object):
    """Pad the multi-view image.
    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",
    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        if self.size is not None:
            padded_img = [impad(
                img, shape=self.size, pad_val=self.pad_val) for img in results['img']]
        elif self.size_divisor is not None:
            padded_img = [impad_to_multiple(
                img, self.size_divisor, pad_val=self.pad_val) for img in results['img']]
        
        results['ori_shape'] = [img.shape for img in results['img']]
        results['img'] = padded_img
        results['img_shape'] = [img.shape for img in padded_img]
        results['pad_shape'] = [img.shape for img in padded_img]
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str

@PIPELINES.register_module()
class ResizeMultiview3D:
    """Resize images & bbox & mask.
    This transform resizes the input image to some scale. Bboxes and masks are
    then resized with the same scale factor. If the input dict contains the key
    "scale", then the scale in the input dict is used, otherwise the specified
    scale in the init method is used. If the input dict contains the key
    "scale_factor" (if MultiScaleFlipAug does not give img_scale but
    scale_factor), the actual scale will be computed by image shape and
    scale_factor.
    `img_scale` can either be a tuple (single-scale) or a list of tuple
    (multi-scale). There are 3 multiscale modes:
    - ``ratio_range is not None``: randomly sample a ratio from the ratio \
      range and multiply it with the image scale.
    - ``ratio_range is None`` and ``multiscale_mode == "range"``: randomly \
      sample a scale from the multiscale range.
    - ``ratio_range is None`` and ``multiscale_mode == "value"``: randomly \
      sample a scale from multiple scales.
    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
        multiscale_mode (str): Either "range" or "value".
        ratio_range (tuple[float]): (min_ratio, max_ratio)
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image.
        bbox_clip_border (bool, optional): Whether to clip the objects outside
            the border of the image. In some dataset like MOT17, the gt bboxes
            are allowed to cross the border of images. Therefore, we don't
            need to clip the gt bboxes in these cases. Defaults to True.
        backend (str): Image resize backend, choices are 'cv2' and 'pillow'.
            These two backends generates slightly different results. Defaults
            to 'cv2'.
        override (bool, optional): Whether to override `scale` and
            `scale_factor` so as to call resize twice. Default False. If True,
            after the first resizing, the existed `scale` and `scale_factor`
            will be ignored so the second resizing can be allowed.
            This option is a work-around for multiple times of resize in DETR.
            Defaults to False.
    """

    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True,
                 bbox_clip_border=True,
                 backend='cv2',
                 override=False):
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            assert is_list_of(self.img_scale, tuple)

        if ratio_range is not None:
            # mode 1: given a scale and a range of image ratio
            assert len(self.img_scale) == 1
        else:
            # mode 2: given multiple scales or a range of scales
            assert multiscale_mode in ['value', 'range']

        self.backend = backend
        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio
        # TODO: refactor the override option in Resize
        self.override = override
        self.bbox_clip_border = bbox_clip_border

    @staticmethod
    def random_select(img_scales):
        """Randomly select an img_scale from given candidates.
        Args:
            img_scales (list[tuple]): Images scales for selection.
        Returns:
            (tuple, int): Returns a tuple ``(img_scale, scale_dix)``, \
                where ``img_scale`` is the selected image scale and \
                ``scale_idx`` is the selected index in the given candidates.
        """

        assert is_list_of(img_scales, tuple)
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        """Randomly sample an img_scale when ``multiscale_mode=='range'``.
        Args:
            img_scales (list[tuple]): Images scale range for sampling.
                There must be two tuples in img_scales, which specify the lower
                and upper bound of image scales.
        Returns:
            (tuple, None): Returns a tuple ``(img_scale, None)``, where \
                ``img_scale`` is sampled scale and None is just a placeholder \
                to be consistent with :func:`random_select`.
        """

        assert is_list_of(img_scales, tuple) and len(img_scales) == 2
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(
            min(img_scale_long),
            max(img_scale_long) + 1)
        short_edge = np.random.randint(
            min(img_scale_short),
            max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        """Randomly sample an img_scale when ``ratio_range`` is specified.
        A ratio will be randomly sampled from the range specified by
        ``ratio_range``. Then it would be multiplied with ``img_scale`` to
        generate sampled scale.
        Args:
            img_scale (tuple): Images scale base to multiply with ratio.
            ratio_range (tuple[float]): The minimum and maximum ratio to scale
                the ``img_scale``.
        Returns:
            (tuple, None): Returns a tuple ``(scale, None)``, where \
                ``scale`` is sampled ratio multiplied with ``img_scale`` and \
                None is just a placeholder to be consistent with \
                :func:`random_select`.
        """

        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, None

    def _random_scale(self, results):
        """Randomly sample an img_scale according to ``ratio_range`` and
        ``multiscale_mode``.
        If ``ratio_range`` is specified, a ratio will be sampled and be
        multiplied with ``img_scale``.
        If multiple scales are specified by ``img_scale``, a scale will be
        sampled according to ``multiscale_mode``.
        Otherwise, single scale will be used.
        Args:
            results (dict): Result dict from :obj:`dataset`.
        Returns:
            dict: Two new keys 'scale` and 'scale_idx` are added into \
                ``results``, which would be used by subsequent pipelines.
        """

        if self.ratio_range is not None:
            scale, scale_idx = self.random_sample_ratio(
                self.img_scale[0], self.ratio_range)
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError
        
        results['scale'] = scale
        results['scale_idx'] = scale_idx

    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""
        # results['scale'] = (1280, 720)
        img_shapes = []
        pad_shapes = []
        scale_factors = []
        keep_ratios = []
        for i in range(len(results['img'])):
            if self.keep_ratio:
                img, scale_factor = imrescale(
                    results['img'][i],
                    results['scale'],
                    return_scale=True,
                    backend=self.backend)
                # the w_scale and h_scale has minor difference
                # a real fix should be done in the mmcv.imrescale in the future
                new_h, new_w = img.shape[:2]
                h, w = results['img'][i].shape[:2]
                w_scale = new_w / w
                h_scale = new_h / h
            else:
                img, w_scale, h_scale = imresize(
                    results['img'][i],
                    results['scale'],
                    return_scale=True,
                    backend=self.backend)
            results['img'][i] = img
            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                dtype=np.float32)
            img_shapes.append(img.shape)
            pad_shapes.append(img.shape)
            scale_factors.append(scale_factor)
            keep_ratios.append(self.keep_ratio)
            #rescale the camera intrinsic
            results['cam_intrinsic'][i][0, 0] *= w_scale 
            results['cam_intrinsic'][i][0, 2] *= w_scale
            results['cam_intrinsic'][i][1, 1] *= h_scale
            results['cam_intrinsic'][i][1, 2] *= h_scale

        results['img_shape'] = img_shapes
        results['pad_shape'] = pad_shapes
        results['scale_factor'] = scale_factors
        results['keep_ratio'] = keep_ratios
        #这里需要进行更改lidar2img的坐标变换，extrinsics =  lidar2cam_rt
        results['lidar2img'] = [results['cam_intrinsic'][i] @ results['lidar2cam'][i] for i in range(len(results['lidar2cam']))]

    def __call__(self, results):
        """Call function to resize images, bounding boxes, masks, semantic
        segmentation map.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor', \
                'keep_ratio' keys are added into result dict.
        """

        if 'scale' not in results:
            self._random_scale(results)
        else:
            if not self.override:
                assert 'scale_factor' not in results, (
                    'scale and scale_factor cannot be both set.')
            else:
                results.pop('scale')
                if 'scale_factor' in results:
                    results.pop('scale_factor')
                self._random_scale(results)

        self._resize_img(results)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(img_scale={self.img_scale}, '
        repr_str += f'multiscale_mode={self.multiscale_mode}, '
        repr_str += f'ratio_range={self.ratio_range}, '
        repr_str += f'keep_ratio={self.keep_ratio}, '
        return repr_str
        
@PIPELINES.register_module()
class NormalizeMultiviewImage(object):
    """Normalize the image.
    Added key is "img_norm_cfg".
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb


    def __call__(self, results):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """

        results['img'] = [imnormalize(img, self.mean, self.std, self.to_rgb) for img in results['img']]
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str


@PIPELINES.register_module()
class PhotoMetricDistortionMultiViewImage:
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, results):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """
        imgs = results['img']
        new_imgs = []
        for img in imgs:
            assert img.dtype == np.float32, \
                'PhotoMetricDistortion needs the input image of dtype np.float32,'\
                ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
            # random brightness
            if random.randint(2):
                delta = random.uniform(-self.brightness_delta,
                                    self.brightness_delta)
                img += delta

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = random.randint(2)
            if mode == 1:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower,
                                        self.contrast_upper)
                    img *= alpha

            # convert color from BGR to HSV
            img = bgr2hsv(img)

            # random saturation
            if random.randint(2):
                img[..., 1] *= random.uniform(self.saturation_lower,
                                            self.saturation_upper)

            # random hue
            if random.randint(2):
                img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
                img[..., 0][img[..., 0] > 360] -= 360
                img[..., 0][img[..., 0] < 0] += 360

            # convert color from HSV to BGR
            img = hsv2bgr(img)

            # random contrast
            if mode == 0:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower,
                                        self.contrast_upper)
                    img *= alpha

            # randomly swap channels
            if random.randint(2):
                img = img[..., random.permutation(3)]
            new_imgs.append(img)
        results['img'] = new_imgs
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(\nbrightness_delta={self.brightness_delta},\n'
        repr_str += 'contrast_range='
        repr_str += f'{(self.contrast_lower, self.contrast_upper)},\n'
        repr_str += 'saturation_range='
        repr_str += f'{(self.saturation_lower, self.saturation_upper)},\n'
        repr_str += f'hue_delta={self.hue_delta})'
        return repr_str



@PIPELINES.register_module()
class CustomCollect3D(object):
    """Collect data from the loader relevant to the specific task.
    This is usually the last stage of the data loader pipeline. Typically keys
    is set to some subset of "img", "proposals", "gt_bboxes",
    "gt_bboxes_ignore", "gt_labels", and/or "gt_masks".
    The "img_meta" item is always populated.  The contents of the "img_meta"
    dictionary depends on "meta_keys". By default this includes:
        - 'img_shape': shape of the image input to the network as a tuple \
            (h, w, c).  Note that images may be zero padded on the \
            bottom/right if the batch tensor is larger than this shape.
        - 'scale_factor': a float indicating the preprocessing scale
        - 'flip': a boolean indicating if image flip transform was used
        - 'filename': path to the image file
        - 'ori_shape': original shape of the image as a tuple (h, w, c)
        - 'pad_shape': image shape after padding
        - 'lidar2img': transform from lidar to image
        - 'depth2img': transform from depth to image
        - 'cam2img': transform from camera to image
        - 'pcd_horizontal_flip': a boolean indicating if point cloud is \
            flipped horizontally
        - 'pcd_vertical_flip': a boolean indicating if point cloud is \
            flipped vertically
        - 'box_mode_3d': 3D box mode
        - 'box_type_3d': 3D box type
        - 'img_norm_cfg': a dict of normalization information:
            - mean: per channel mean subtraction
            - std: per channel std divisor
            - to_rgb: bool indicating if bgr was converted to rgb
        - 'pcd_trans': point cloud transformations
        - 'sample_idx': sample index
        - 'pcd_scale_factor': point cloud scale factor
        - 'pcd_rotation': rotation applied to point cloud
        - 'pts_filename': path to point cloud file.
    Args:
        keys (Sequence[str]): Keys of results to be collected in ``data``.
        meta_keys (Sequence[str], optional): Meta keys to be converted to
            ``mmcv.DataContainer`` and collected in ``data[img_metas]``.
            Default: ('filename', 'ori_shape', 'img_shape', 'lidar2img',
            'depth2img', 'cam2img', 'pad_shape', 'scale_factor', 'flip',
            'pcd_horizontal_flip', 'pcd_vertical_flip', 'box_mode_3d',
            'box_type_3d', 'img_norm_cfg', 'pcd_trans',
            'sample_idx', 'pcd_scale_factor', 'pcd_rotation', 'pts_filename')
    """

    def __init__(self,
                 keys,
                 meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                            'depth2img', 'cam2img', 'pad_shape', 'gt_bboxes_3d','gt_labels_3d',
                            'scale_factor', 'flip', 'pcd_horizontal_flip',
                            'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                            'img_norm_cfg', 'pcd_trans', 'sample_idx', 'prev_idx', 'next_idx',
                            'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                            'transformation_3d_flow', 'scene_token',
                            'can_bus','folder','frame_idx','vlm_labels', 'lidar2ego',
                            'traffic_state_mask', 'traffic_state',
                            )):
        # TODO(yzj) bevformer meta_keys has lidar2cam
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        """Call function to collect keys in results. The keys in ``meta_keys``
        will be converted to :obj:`mmcv.DataContainer`.
        Args:
            results (dict): Result dict contains the data to collect.
        Returns:
            dict: The result dict contains the following keys
                - keys in ``self.keys``
                - ``img_metas``
        """
       
        data = {}
        img_metas = {}
        for key in self.meta_keys:
            if key in results:
                img_metas[key] = results[key]

        data['img_metas'] = DC(img_metas, cpu_only=True)
        for key in self.keys:
            data[key] = results[key]
        return data

    def __repr__(self):
        """str: Return a string that describes the module."""
        return self.__class__.__name__ + \
            f'(keys={self.keys}, meta_keys={self.meta_keys})'



@PIPELINES.register_module()
class RandomScaleImageMultiViewImage(object):
    """Random scale the image
    Args:
        scales
    """

    def __init__(self, scales=[]):
        self.scales = scales
        assert len(self.scales)==1

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        rand_ind = np.random.permutation(range(len(self.scales)))[0]
        rand_scale = self.scales[rand_ind]

        y_size = [int(img.shape[0] * rand_scale) for img in results['img']]
        x_size = [int(img.shape[1] * rand_scale) for img in results['img']]
        scale_factor = np.eye(4)
        scale_factor[0, 0] *= rand_scale
        scale_factor[1, 1] *= rand_scale
        results['img'] = [imresize(img, (x_size[idx], y_size[idx]), return_scale=False) for idx, img in
                          enumerate(results['img'])]
        lidar2img = [scale_factor @ l2i for l2i in results['lidar2img']]
        results['lidar2img'] = lidar2img
        results['img_shape'] = [img.shape for img in results['img']]
        results['ori_shape'] = [img.shape for img in results['img']]

        return results


    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.scales}, '
        return repr_str

@PIPELINES.register_module()
class ObjectRangeFilterTrack(object):
    """Filter objects by the range.
    Args:
        point_cloud_range (list[float]): Point cloud range.
    """

    def __init__(self, point_cloud_range):
        self.pcd_range = np.array(point_cloud_range, dtype=np.float32)

    def __call__(self, input_dict):
        """Call function to filter objects by the range.
        Args:
            input_dict (dict): Result dict from loading pipeline.
        Returns:
            dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d' \
                keys are updated in the result dict.
        """
        # Check points instance type and initialise bev_range
        if isinstance(input_dict['gt_bboxes_3d'],
                      (LiDARInstance3DBoxes, DepthInstance3DBoxes)):
            bev_range = self.pcd_range[[0, 1, 3, 4]]
        elif isinstance(input_dict['gt_bboxes_3d'], CameraInstance3DBoxes):
            bev_range = self.pcd_range[[0, 2, 3, 5]]

        if 'gt_inds' in input_dict['ann_info'].keys():
            input_dict['gt_inds'] = input_dict['ann_info']['gt_inds']
        if 'gt_fut_traj' in input_dict['ann_info'].keys():
            input_dict['gt_fut_traj'] = input_dict['ann_info']['gt_fut_traj']
        if 'gt_fut_traj_mask' in input_dict['ann_info'].keys():
            input_dict['gt_fut_traj_mask'] = input_dict['ann_info']['gt_fut_traj_mask']
        if 'gt_past_traj' in input_dict['ann_info'].keys():
            input_dict['gt_past_traj'] = input_dict['ann_info']['gt_past_traj']
        if 'gt_past_traj_mask' in input_dict['ann_info'].keys():
            input_dict['gt_past_traj_mask'] = input_dict['ann_info']['gt_past_traj_mask']
        if 'gt_sdc_bbox' in input_dict['ann_info'].keys():
            input_dict['gt_sdc_bbox'] = input_dict['ann_info']['gt_sdc_bbox']
            input_dict['gt_sdc_label'] = input_dict['ann_info']['gt_sdc_label']
            input_dict['gt_sdc_fut_traj'] = input_dict['ann_info']['gt_sdc_fut_traj']
            input_dict['gt_sdc_fut_traj_mask'] = input_dict['ann_info']['gt_sdc_fut_traj_mask']

        gt_bboxes_3d = input_dict['gt_bboxes_3d']
        gt_labels_3d = input_dict['gt_labels_3d']
        gt_inds = input_dict['gt_inds']
        gt_fut_traj = input_dict['gt_fut_traj']
        gt_fut_traj_mask = input_dict['gt_fut_traj_mask']
        gt_past_traj = input_dict['gt_past_traj']
        gt_past_traj_mask = input_dict['gt_past_traj_mask']

        mask = gt_bboxes_3d.in_range_bev(bev_range)
        gt_bboxes_3d = gt_bboxes_3d[mask]
        # mask is a torch tensor but gt_labels_3d is still numpy array
        # using mask to index gt_labels_3d will cause bug when
        # len(gt_labels_3d) == 1, where mask=1 will be interpreted
        # as gt_labels_3d[1] and cause out of index error
        mask = mask.numpy().astype(np.bool)
        gt_labels_3d = gt_labels_3d[mask]
        gt_inds = gt_inds[mask]
        gt_fut_traj = gt_fut_traj[mask]
        gt_fut_traj_mask = gt_fut_traj_mask[mask]
        gt_past_traj = gt_past_traj[mask]
        gt_past_traj_mask = gt_past_traj_mask[mask]

        # limit rad to [-pi, pi]
        gt_bboxes_3d.limit_yaw(offset=0.5, period=2 * np.pi)
        input_dict['gt_bboxes_3d'] = gt_bboxes_3d
        input_dict['gt_labels_3d'] = gt_labels_3d
        input_dict['gt_inds'] = gt_inds
        input_dict['gt_fut_traj'] = gt_fut_traj
        input_dict['gt_fut_traj_mask'] = gt_fut_traj_mask
        input_dict['gt_past_traj'] = gt_past_traj
        input_dict['gt_past_traj_mask'] = gt_past_traj_mask
        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(point_cloud_range={self.pcd_range.tolist()})'
        return repr_str

@PIPELINES.register_module()
class ObjectNameFilterTrack(object):
    """Filter GT objects by their names.
    Args:
        classes (list[str]): List of class names to be kept for training.
    """

    def __init__(self, classes):
        self.classes = classes
        self.labels = list(range(len(self.classes)))

    def __call__(self, input_dict):
        """Call function to filter objects by their names.
        Args:
            input_dict (dict): Result dict from loading pipeline.
        Returns:
            dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d' \
                keys are updated in the result dict.
        """
        gt_labels_3d = input_dict['gt_labels_3d']
        gt_bboxes_mask = np.array([n in self.labels for n in gt_labels_3d],
                                  dtype=np.bool_)
        input_dict['gt_bboxes_3d'] = input_dict['gt_bboxes_3d'][gt_bboxes_mask]
        input_dict['gt_labels_3d'] = input_dict['gt_labels_3d'][gt_bboxes_mask]
        input_dict['gt_inds'] = input_dict['gt_inds'][gt_bboxes_mask]
        input_dict['gt_fut_traj'] = input_dict['gt_fut_traj'][gt_bboxes_mask]
        input_dict['gt_fut_traj_mask'] = input_dict['gt_fut_traj_mask'][gt_bboxes_mask]
        input_dict['gt_past_traj'] = input_dict['gt_past_traj'][gt_bboxes_mask]
        input_dict['gt_past_traj_mask'] = input_dict['gt_past_traj_mask'][gt_bboxes_mask]
        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(classes={self.classes})'
        return repr_str

@PIPELINES.register_module()
class CustomObjectRangeFilter(ObjectRangeFilter):
    def __call__(self, results):
        """Call function to filter objects by the range.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d'
                keys are updated in the result dict.
        """
        # Check points instance type and initialise bev_range
        if isinstance(results['gt_bboxes_3d'],
                        (LiDARInstance3DBoxes, DepthInstance3DBoxes)):
            bev_range = self.pcd_range[[0, 1, 3, 4]]
        elif isinstance(results['gt_bboxes_3d'], CameraInstance3DBoxes):
            bev_range = self.pcd_range[[0, 2, 3, 5]]

        gt_bboxes_3d = results['gt_bboxes_3d']
        gt_labels_3d = results['gt_labels_3d']
        mask = gt_bboxes_3d.in_range_bev(bev_range)
        gt_bboxes_3d = gt_bboxes_3d[mask]
        # mask is a torch tensor but gt_labels_3d is still numpy array
        # using mask to index gt_labels_3d will cause bug when
        # len(gt_labels_3d) == 1, where mask=1 will be interpreted
        # as gt_labels_3d[1] and cause out of index error
        gt_labels_3d = gt_labels_3d[mask.numpy().astype(np.bool)]

        # limit rad to [-pi, pi]
        gt_bboxes_3d.limit_yaw(offset=0.5, period=2 * np.pi)
        results['gt_bboxes_3d'] = gt_bboxes_3d
        results['gt_labels_3d'] = gt_labels_3d
        # results['ann_tokens'] = results['ann_tokens'][mask.numpy().astype(np.bool)]

        return results

@PIPELINES.register_module()
class CustomObjectNameFilter(ObjectNameFilter):
    def __call__(self, results):
        """Call function to filter objects by their names.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d'
                keys are updated in the result dict.
        """
        gt_labels_3d = results['gt_labels_3d']
        gt_bboxes_mask = np.array([n in self.labels for n in gt_labels_3d],
                                  dtype=np.bool_)
        results['gt_bboxes_3d'] = results['gt_bboxes_3d'][gt_bboxes_mask]
        results['gt_labels_3d'] = results['gt_labels_3d'][gt_bboxes_mask]
        # results['ann_tokens'] = results['ann_tokens'][gt_bboxes_mask]

        return results


@PIPELINES.register_module()
class VADObjectRangeFilter(object):
    """Filter objects by the range, and also filter corresponding fut trajs

    Args:
        point_cloud_range (list[float]): Point cloud range.
    """

    def __init__(self, point_cloud_range):
        self.pcd_range = np.array(point_cloud_range, dtype=np.float32)

    def __call__(self, input_dict):
        """Call function to filter objects by the range.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d' \
                keys are updated in the result dict.
        """
        # Check points instance type and initialise bev_range
        if isinstance(input_dict['gt_bboxes_3d'],
                      (LiDARInstance3DBoxes, DepthInstance3DBoxes)):
            bev_range = self.pcd_range[[0, 1, 3, 4]]
        elif isinstance(input_dict['gt_bboxes_3d'], CameraInstance3DBoxes):
            bev_range = self.pcd_range[[0, 2, 3, 5]]

        gt_bboxes_3d = input_dict['gt_bboxes_3d']
        gt_labels_3d = input_dict['gt_labels_3d']
        
        
        mask = gt_bboxes_3d.in_range_bev(bev_range)
        gt_bboxes_3d = gt_bboxes_3d[mask]
        # mask is a torch tensor but gt_labels_3d is still numpy array
        # using mask to index gt_labels_3d will cause bug when
        # len(gt_labels_3d) == 1, where mask=1 will be interpreted
        # as gt_labels_3d[1] and cause out of index error
        gt_labels_3d = gt_labels_3d[mask.numpy().astype(np.bool)]
        if 'traffic_state_mask' in input_dict:
            gt_traffic_state = input_dict['traffic_state']
            gt_traffic_state_mask = input_dict['traffic_state_mask']
            gt_traffic_state = gt_traffic_state[mask.numpy().astype(np.bool)]
            gt_traffic_state_mask = gt_traffic_state_mask[mask.numpy().astype(np.bool)]
            input_dict['traffic_state'] = gt_traffic_state 
            input_dict['traffic_state_mask'] = gt_traffic_state_mask

        

        # limit rad to [-pi, pi]
        gt_bboxes_3d.limit_yaw(offset=0.5, period=2 * np.pi)
        input_dict['gt_bboxes_3d'] = gt_bboxes_3d
        input_dict['gt_labels_3d'] = gt_labels_3d

        if 'attr_labels' in input_dict:
            gt_attr_labels = input_dict['attr_labels']
            gt_attr_labels = gt_attr_labels[mask.numpy().astype(np.bool)]
            input_dict['gt_attr_labels'] = gt_attr_labels

        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(point_cloud_range={self.pcd_range.tolist()})'
        return repr_str


@PIPELINES.register_module()
class VADObjectNameFilter(object):
    """Filter GT objects by their names, , and also filter corresponding fut trajs

    Args:
        classes (list[str]): List of class names to be kept for training.
    """

    def __init__(self, classes):
        self.classes = classes
        self.labels = list(range(len(self.classes)))

    def __call__(self, input_dict):
        """Call function to filter objects by their names.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d' \
                keys are updated in the result dict.
        """
        gt_labels_3d = input_dict['gt_labels_3d']
        gt_bboxes_mask = np.array([n in self.labels for n in gt_labels_3d],
                                  dtype=np.bool_)
        input_dict['gt_bboxes_3d'] = input_dict['gt_bboxes_3d'][gt_bboxes_mask]
        input_dict['gt_labels_3d'] = input_dict['gt_labels_3d'][gt_bboxes_mask]
        if 'gt_attr_labels' in input_dict:
            input_dict['gt_attr_labels'] = input_dict['gt_attr_labels'][gt_bboxes_mask]
        if 'traffic_state_mask' in input_dict:
            input_dict['traffic_state_mask'] = input_dict['traffic_state_mask'][gt_bboxes_mask]
            input_dict['traffic_state'] = input_dict['traffic_state'][gt_bboxes_mask]
        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(classes={self.classes})'
        return repr_str

@PIPELINES.register_module()
class CustomPointsRangeFilter:
    """Filter points by the range.
    Args:
        point_cloud_range (list[float]): Point cloud range.
    """

    def __init__(self, point_cloud_range):
        self.pcd_range = np.array(point_cloud_range, dtype=np.float32)

    def __call__(self, data):
        """Call function to filter points by the range.
        Args:
            data (dict): Result dict from loading pipeline.
        Returns:
            dict: Results after filtering, 'points', 'pts_instance_mask' \
                and 'pts_semantic_mask' keys are updated in the result dict.
        """
        points = data["points"]
        points_mask = points.in_range_3d(self.pcd_range)
        clean_points = points[points_mask]
        data["points"] = clean_points
        return data

def format_number(n, decimal_places=1):
    if abs(round(n, decimal_places)) <= 1e-2:
         return 0.0
    else:
        format_string = f"{{n:+.{decimal_places}f}}"
        return format_string.format(n=n)   
def post_process_coords(corner_coords, imsize=(1600, 900)):
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = canvas_box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)

        if isinstance(img_intersection, Polygon):
            intersection_coords = np.array([coord for coord in img_intersection.exterior.coords])
            
            # 计算 min_x, min_y, max_x, max_y
            min_x = min(intersection_coords[:, 0])
            min_y = min(intersection_coords[:, 1])
            max_x = max(intersection_coords[:, 0])
            max_y = max(intersection_coords[:, 1])

            return min_x, min_y, max_x, max_y
        else:
            return None
    else:
        return None
    
def analyze_position(x, y, angle_deg):
    direction = ''
    if x > 0:
        direction += 'front'
    elif x < 0:
        direction += 'back'

    if y > 2.5:
        direction += ' left'
    elif y < -2.5:
        direction += ' right'

    
    if abs(angle_deg) < 45:
        direction += ", same direction as you, "
    elif abs(abs(angle_deg) - 180) < 45:
        direction += ", opposite direction from you, "
    elif abs(angle_deg - 90) < 45:
        direction += ", heading from right to left, "
    elif abs(angle_deg + 90) < 45:
        direction += ", heading from left to right, "

    return direction.strip()



@PIPELINES.register_module()
class LoadAnnoatationVQA():
    def __init__(
            self, 
            tokenizer, 
            max_length, 
            base_desc_path=None,
            n_gen=1, 
            planning_qa_only=False,
            planning_qa_last=False,
            use_gen_token=False,
            use_meta_action=False,
            pretrain = False,
            planning_qa_ratio=0.8,
            mix_qa_training=False,
            is_decoupling=False,
            conv_template = None,
            ):
        
        self.tokenizer =  AutoTokenizer.from_pretrained(tokenizer,
                                            model_max_length=max_length,
                                            padding_side="right",
                                            use_fast=False,
                                            )
        self.n_gen = n_gen
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.planning_qa_only = planning_qa_only
        self.planning_qa_last = planning_qa_last
        self.base_desc_path = base_desc_path
        self.mix_qa_training = mix_qa_training
        self.planning_qa_ratio = planning_qa_ratio
        self.r_random_generator = np.random.default_rng(seed=42)
        self.shuffle_random_generator = np.random.default_rng(seed=99)
        CLASSES = ('car','van','truck','bicycle','traffic_sign','traffic_cone','traffic_light','pedestrian','others')
        self.id2cat = {i: name for i, name in enumerate(CLASSES)}
        self.side = {
        'singapore': 'left',
        'boston': 'right',
        }
       
        self.use_gen_token = use_gen_token
        self.is_decoupling =is_decoupling
        self.use_meta_action=use_meta_action
        self.pretrain = pretrain
        self.conv_template = conv_template

        self.SPEED_MAPPING = {
        '<maintain_moderate_speed>': 0,
        '<stop>':1,
        '<maintain_slow_speed>':2,
        '<speed_up>':3,
        '<slow_down>':4,
        '<maintain_fast_speed>':5,
        '<slow_down_rapidly>':6}

        self.PATH_MAPPING= {
        '<lanefollow>': 0,
        '<straight>':1,
        '<turn_left>': 2,
        '<change_lane_left>':3,
        '<turn_right>': 4,
        '<change_lane_right>':5, 
        }
        if self.conv_template is not None:
            conversation_lib.default_conversation = conversation_lib.conv_templates[self.conv_template]
        
        if self.conv_template == 'llava_qwen2':
            self.tokenizer.eos_token_id = self.tokenizer.convert_tokens_to_ids('<|im_end|>')
        else:
            self.tokenizer.pad_token = self.tokenizer.unk_token
        

    def preprocess_vqa(self, results):
        sources = []
        if self.base_desc_path is not None:
            image_path = Path(results['img_filename'][0])
            json_directory = image_path.parent.parent.parent.stem 

            with open(self.base_desc_path+'/'+json_directory +'/'+ f'{image_path.stem}.json', 'r') as f:
                desc = json.load(f)
            sources.extend(desc)
        return sources  
    
    def online_vqa(self, results):
        sources = []
        gt_bboxes_2d = []
        gt_bboxes_3d = copy.deepcopy(results['gt_bboxes_3d'])
        gt_bboxes_3d_points = gt_bboxes_3d.corners   
        gt_bboxes_points = gt_bboxes_3d_points.view(-1, 3)
        gt_bboxes_points = np.concatenate((gt_bboxes_points[:, :3], np.ones(gt_bboxes_points.shape[0])[:, None]), axis=1)
            
        if len(gt_bboxes_3d) >= 1:
            centers = torch.FloatTensor(max(self.n_gen, len(gt_bboxes_3d)), 2).uniform_(0, 20) 
            bbox_center = gt_bboxes_3d.center[:, :2] + 5 * (torch.rand_like(gt_bboxes_3d.center[:, :2]) * 2 - 1)
            centers = torch.cat([bbox_center, centers], dim=0)
            indices = torch.randperm(centers.size(0))[:self.n_gen]
            centers = centers[indices]

            for center in centers:
                objs_near = []
                for i in range(len(gt_bboxes_3d)):
                    gt_box = gt_bboxes_3d[i]
                    dis = torch.norm(gt_box.center[0, :2] - center)
                    if dis < 10:
                        objs_near.append(self.format_det_answer(i, gt_bboxes_3d, results))
                if len(objs_near) == 0:
                    answer = f"There are no objects nearby."
                else:
                    answer = "There are the following objects nearby:"
                    answer += ' '.join(objs_near)
                sources.append(
                [
                    {"from": 'human',
                    "value": f"What objects are there near the position ({format_number(center[0].item())}, {format_number(center[1].item())})?"},
                    {"from": 'gpt',
                    "value": f"{answer}",}
                    ]
            )
            
        return sources
    
    def format_det_answer(self, index, gt_bboxes_3d, results):
        x = gt_bboxes_3d.tensor[index][0].item()
        y = gt_bboxes_3d.tensor[index][1].item()
        z = gt_bboxes_3d.tensor[index][2].item()
        l = gt_bboxes_3d.tensor[index][3].item()
        w = gt_bboxes_3d.tensor[index][4].item()
        h = gt_bboxes_3d.tensor[index][5].item()
        yaw = gt_bboxes_3d.tensor[index][6].item()
        vx = gt_bboxes_3d.tensor[index][7].item()
        vy = gt_bboxes_3d.tensor[index][8].item()
        yaw = math.degrees(yaw)
        position = analyze_position(x, y, yaw)
        answer = f"{self.id2cat[results['gt_labels_3d'][index]]} in the {position} "
        answer += f"location: <{format_number(x)}, {format_number(y)}>, " 
        answer += f"length: {l:.1f}, width: {w:.1f}, height: {h:.1f}, "
        answer += f"angles in degrees: {format_number(yaw)}"
        if np.sqrt(vx**2 + vy**2) > 0.2:
            answer += f", velocity: <{format_number(vx)}, {format_number(vy)}>.  "  
        else:
            answer += "."

        return answer

    def __call__(self, results):
        traj = None
        prompt = f"You are driving a car."
        sources= []


        if self.use_meta_action:
            meta_action_qa = self.preprocess_vqa(results)[-1]# The last is meta action qa
            sources.append(meta_action_qa)
            meta_action = meta_action_qa[-1]['value'].split(' ')
            speed = meta_action[-2] 
            path = meta_action[-1]   
            cmd_speed = self.command2hot(self.SPEED_MAPPING.get(speed), 7)
            cmd_path = self.command2hot(self.PATH_MAPPING.get(path), 6)
            results['cmd_speed'] = cmd_speed
            results['cmd_path'] = cmd_path

        if not self.planning_qa_only:
            sources += self.preprocess_vqa(results)[:-1]
            online_sources = self.online_vqa(results)  
            sources += online_sources
            random.shuffle(sources)  

        if self.use_gen_token:
            planning_qa = [
                [{"from": 'human',
                "value": "Based on the above information, please provide a safe, executable, and reasonable planning trajectory for the ego car."},
                {"from": 'gpt',
                "value": "Here is the planning trajectory <waypoint_ego>"}]
            ]
            
            if self.is_decoupling:
                planning_qa = [
                    [{"from": 'human',
                    "value": "Based on the above information, please provide a safe, executable, and reasonable planning trajectory for the ego car."},
                    {"from": 'gpt',
                    "value": "Here is the planning trajectory <waypoint_ego> <path_waypoint_ego>"}]
                ]
        
        if not self.pretrain:
            if self.mix_qa_training:
                r = self.r_random_generator.uniform()
                if r < self.planning_qa_ratio:
                    if self.use_meta_action:
                        sources =[]
                        sources.append(meta_action_qa)
                    else:
                        sources =[]
                    if self.planning_qa_last:
                        sources += planning_qa
                    else:
                        sources = planning_qa + sources
                else:
                    self.shuffle_random_generator.shuffle(sources) 
            else:
                if self.planning_qa_last:
                    sources += planning_qa
                else:
                    sources = planning_qa + sources
  
        vqa_anno = [item for pair in sources for item in pair]

        if self.use_meta_action:
            new_speed_tokens = ["<maintain_moderate_speed>", "<stop>", "<maintain_slow_speed>", "<speed_up>",'<slow_down>','<maintain_fast_speed>','<slow_down_rapidly>']
            new_path_tokens = ["<lanefollow>", '<straight>', '<turn_left>','<change_lane_left>','<turn_right>','<change_lane_right>']
            num_new_tokens = self.tokenizer.add_tokens(new_speed_tokens, special_tokens = True)            
            num_new_tokens = self.tokenizer.add_tokens(new_path_tokens, special_tokens = True)  

        if self.use_gen_token:
            num_new_tokens = self.tokenizer.add_tokens(["<waypoint_ego>"], special_tokens = True)
            if self.is_decoupling:
                num_new_tokens = self.tokenizer.add_tokens(["<path_waypoint_ego>"], special_tokens = True)
 
        vqa_anno[0]['value'] = DEFAULT_IMAGE_TOKEN + '\n' + prompt + vqa_anno[0]['value']  

        if self.conv_template == 'llava_qwen2':
            vqa_converted = preprocess_qwen2([vqa_anno], self.tokenizer, True)
        else:
            vqa_converted = preprocess([vqa_anno], self.tokenizer, True)

        input_ids = vqa_converted['input_ids'][0]
        vlm_labels = vqa_converted['labels'][0] 
        if not self.pretrain:
           if self.planning_qa_last and (len(vqa_converted['input_ids'][0]) == self.tokenizer.model_max_length):
                print('Token indices sequence length is too long, only basic planning QA is reserved.')
                sources = planning_qa 
                vqa_anno = [item for pair in sources for item in pair]
                vqa_anno[0]['value'] = DEFAULT_IMAGE_TOKEN + '\n' + prompt + vqa_anno[0]['value']  
                
                if self.conv_template == 'llava_qwen2':
                    vqa_converted = preprocess_qwen2([vqa_anno], self.tokenizer, True)
                else:
                    vqa_converted = preprocess([vqa_anno], self.tokenizer, True)

                input_ids = vqa_converted['input_ids'][0]
                vlm_labels = vqa_converted['labels'][0]


        vlm_attn_mask = torch.ones(input_ids.shape).to(dtype=torch.int)
        results['input_ids'] = input_ids
        results['vlm_labels'] = vlm_labels
        results['vlm_attn_mask'] = vlm_attn_mask

        return results
    
    def command2hot(self,command,max_dim=6):
        cmd_one_hot = np.zeros(max_dim)
        cmd_one_hot[command] = 1
        return cmd_one_hot     
        
    def remove_object_numbers(self, text): # for clear data
        pattern = f'\s\(object \d+\)' 
        cleaned_text = re.sub(pattern, '', text)
        return cleaned_text

@PIPELINES.register_module()
class ResizeCropFlipRotImage():
    def __init__(self, data_aug_conf=None, with_2d=False, filter_invisible=True, training=True):
        self.data_aug_conf = data_aug_conf
        self.training = training
        self.min_size = 2.0
        self.with_2d = with_2d
        self.filter_invisible = filter_invisible

    def __call__(self, results):

        imgs = results['img']
        N = len(imgs)
        new_imgs = []
        assert self.data_aug_conf['rot_lim'] == (0.0, 0.0), "Rotation is not currently supported"
        resize, resize_dims, crop, flip, rotate = self._sample_augmentation()

        for i in range(N):
            img = Image.fromarray(np.uint8(imgs[i]))
            img, ida_mat = self._img_transform(
                img,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate,
            )

            new_imgs.append(np.array(img).astype(np.float32))
            results['cam_intrinsic'][i][:3, :3] = ida_mat @ results['cam_intrinsic'][i][:3, :3]

        results['img'] = new_imgs
        results['lidar2img'] = [results['cam_intrinsic'][i] @ results['lidar2cam'][i] for i in range(len(results['lidar2cam']))]

        return results

    def _bboxes_transform(self, bboxes, centers2d, gt_labels, depths,resize, crop, flip):
        assert len(bboxes) == len(centers2d) == len(gt_labels) == len(depths)
        fH, fW = self.data_aug_conf["final_dim"]
        bboxes = bboxes * resize
        bboxes[:, 0] = bboxes[:, 0] - crop[0]
        bboxes[:, 1] = bboxes[:, 1] - crop[1]
        bboxes[:, 2] = bboxes[:, 2] - crop[0]
        bboxes[:, 3] = bboxes[:, 3] - crop[1]
        bboxes[:, 0] = np.clip(bboxes[:, 0], 0, fW)
        bboxes[:, 2] = np.clip(bboxes[:, 2], 0, fW)
        bboxes[:, 1] = np.clip(bboxes[:, 1], 0, fH) 
        bboxes[:, 3] = np.clip(bboxes[:, 3], 0, fH)
        keep = ((bboxes[:, 2] - bboxes[:, 0]) >= self.min_size) & ((bboxes[:, 3] - bboxes[:, 1]) >= self.min_size)


        if flip:
            x0 = bboxes[:, 0].copy()
            x1 = bboxes[:, 2].copy()
            bboxes[:, 2] = fW - x0
            bboxes[:, 0] = fW - x1
        bboxes = bboxes[keep]

        centers2d  = centers2d * resize
        centers2d[:, 0] = centers2d[:, 0] - crop[0]
        centers2d[:, 1] = centers2d[:, 1] - crop[1]
        centers2d[:, 0] = np.clip(centers2d[:, 0], 0, fW)
        centers2d[:, 1] = np.clip(centers2d[:, 1], 0, fH) 
        if flip:
            centers2d[:, 0] = fW - centers2d[:, 0]

        centers2d = centers2d[keep]
        gt_labels = gt_labels[keep]
        depths = depths[keep]

        return bboxes, centers2d, gt_labels, depths


    def _filter_invisible(self, bboxes, centers2d, gt_labels, depths):
        # filter invisible 2d bboxes
        assert len(bboxes) == len(centers2d) == len(gt_labels) == len(depths)
        fH, fW = self.data_aug_conf["final_dim"]
        indices_maps = np.zeros((fH,fW))
        tmp_bboxes = np.zeros_like(bboxes)
        tmp_bboxes[:, :2] = np.ceil(bboxes[:, :2])
        tmp_bboxes[:, 2:] = np.floor(bboxes[:, 2:])
        tmp_bboxes = tmp_bboxes.astype(np.int64)
        sort_idx = np.argsort(-depths, axis=0, kind='stable')
        tmp_bboxes = tmp_bboxes[sort_idx]
        bboxes = bboxes[sort_idx]
        depths = depths[sort_idx]
        centers2d = centers2d[sort_idx]
        gt_labels = gt_labels[sort_idx]
        for i in range(bboxes.shape[0]):
            u1, v1, u2, v2 = tmp_bboxes[i]
            indices_maps[v1:v2, u1:u2] = i
        indices_res = np.unique(indices_maps).astype(np.int64)
        bboxes = bboxes[indices_res]
        depths = depths[indices_res]
        centers2d = centers2d[indices_res]
        gt_labels = gt_labels[indices_res]

        return bboxes, centers2d, gt_labels, depths



    def _get_rot(self, h):
        return torch.Tensor(
            [
                [np.cos(h), np.sin(h)],
                [-np.sin(h), np.cos(h)],
            ]
        )

    def _img_transform(self, img, resize, resize_dims, crop, flip, rotate):
        ida_rot = torch.eye(2)
        ida_tran = torch.zeros(2)
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # post-homography transformation
        ida_rot *= resize
        ida_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            ida_rot = A.matmul(ida_rot)
            ida_tran = A.matmul(ida_tran) + b
        A = self._get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        ida_rot = A.matmul(ida_rot)
        ida_tran = A.matmul(ida_tran) + b
        ida_mat = torch.eye(3)
        ida_mat[:2, :2] = ida_rot
        ida_mat[:2, 2] = ida_tran
        return img, ida_mat

    def _sample_augmentation(self):
        H, W = self.data_aug_conf["H"], self.data_aug_conf["W"]
        fH, fW = self.data_aug_conf["final_dim"]
        if self.training:
            resize = np.random.uniform(*self.data_aug_conf["resize_lim"])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_aug_conf["bot_pct_lim"])) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf["rand_flip"] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf["rot_lim"])
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_aug_conf["bot_pct_lim"])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate


@PIPELINES.register_module()
class LoadAnnoatationCriticalVQATest():
    def __init__(
            self, 
            tokenizer, 
            max_length,
            load_type=["conv", "planning", "counter"], 
            use_token=False,
            use_multi_modal_special_token=False,
            planning_qa_command=False,
            desc_qa=False,
            use_gen_token=False,
            merge_multiround_qa_into_one=False,
            ):
        self.tokenizer =  AutoTokenizer.from_pretrained(tokenizer,
                                            model_max_length=max_length,
                                            padding_side="right",
                                            use_fast=False,
                                            )
        # self.tokenizer.pad_token = self.tokenizer.unk_token
        self.load_type = load_type
        self.use_token = use_token
        self.use_multi_modal_special_token = use_multi_modal_special_token
        self.side = {
        'singapore': 'left',
        'boston': 'right',
    }
        self.template = [
                        "What can you tell about the current driving conditions from the images?",
                        "What can be observed in the panoramic images provided?",
                        "Can you provide a summary of the current driving scenario based on the input images?",
                        "What can you observe from the provided images regarding the driving conditions?",
                        "Please describe the current driving conditions based on the images provided.",
                        "Can you describe the current weather conditions and the general environment depicted in the images?",
                        "Please describe the current driving conditions based on the input images.",
                        "Could you summarize the current driving conditions based on the input images?",
                        "Please provide an overview of the current driving conditions based on the images.",
                        "Can you summarize what the panoramic images show?",
                        "Can you describe the overall conditions and environment based on the images?",
                        "Could you describe the overall environment and objects captured in the images provided?"
                        ]

        self.critical_object_template = [
                        "Where are the critical objects in the scene and what impact do they have on the ego vehicle?",
                        "Identify the significant objects in the scene and their specific impacts on the ego vehicle.",
                        "Can you pinpoint the critical objects in the scene and describe their influence on the ego vehicle?",
                        "Which objects in the scene are critical, and what effects do they have on the ego vehicle's movement?",
                        "Please describe the critical objects in the scene, their positions, and the influence they have on the ego vehicle."
                        ]

        self.command_template = [
                                "The current driving instruction is to turn left.",
                                "The current driving instruction is to turn right.",
                                "The current driving instruction is to go straight.",
                                "The current driving instruction is to drive following the lane.",
                                "The current driving instruction is to change lanes to the left.",
                                "The current driving instruction is to change lanes to the right."]

        if self.use_multi_modal_special_token:
            assert self.use_token
            self.special_token_list = [
                                        "<left_waypoint>",
                                        "<right_waypoint>",
                                        "<straight_waypoint>",
                                        "<follow_waypoint>",
                                        "<change_left_waypoint>",
                                        "<change_right_waypoint>"
                                        ]
        self.planning_qa_command = planning_qa_command
        self.desc_qa = desc_qa
        self.use_gen_token = use_gen_token
        if self.use_gen_token:
            assert not self.use_token
        self.merge_multiround_qa_into_one = merge_multiround_qa_into_one
        self.merge_qa_prompt = ['I will ask you three questions, and you need to answer them one by one.',
                                'The first question is: ',
                                'The second question is: ',
                                'The third question is: ',
                                ]
        
    def preprocess_vqa(self, results):
        sources = []
        question = str(random.choice(self.template))
        critical_object_question = str(random.choice(self.critical_object_template))
        if "critical_qa" in self.load_type:
                sources.append(
                    [
                        {"from": 'human',
                        "value": question},
                        {"from": 'gpt',
                        "value": ""}
                        ]
                )
                sources.append(
                            [
                                {"from": 'human',
                                "value": critical_object_question},
                                {"from": 'gpt',
                                "value": ""}
                                ]
                        )
                sources.append(
                        [
                            {"from": 'human',
                            "value": "Please describe your driving behavior and explain the reasons."},
                            {"from": 'gpt',
                            "value": ""}
                            ]
                    )
                if self.merge_multiround_qa_into_one:
                    sources = []
                    sources.append(
                            [
                                {"from": 'human',
                                "value": self.merge_qa_prompt[0] + ' ' + self.merge_qa_prompt[1] + question + ' ' + self.merge_qa_prompt[2] + critical_object_question + ' ' + self.merge_qa_prompt[3] + "Please describe your driving behavior and explain the reasons."},
                                {"from": 'gpt',
                                "value": ""}
                                ]
                        )
        if "planning" in self.load_type: # planning trajs
            sources.append(
                    [
                        {"from": 'human',
                        "value": "Please provide the planning trajectory for the ego car without reasons."},
                        {"from": 'gpt',
                        "value": ""}
                        ]
                )
        if "short" in self.load_type: # short driving action
            sources.append(
                    [
                        {"from": 'human',
                        "value": "Please shortly describe your driving action."},
                        {"from": 'gpt',
                        "value": ""}
                        ]
                )
        if "conv" in self.load_type: # conversation
            question = str(random.choice(self.template)) # detailed description
            sources.append(
                        [
                            {"from": 'human',
                            "value": question},
                            {"from": 'gpt',
                            "value": ""}
                            ]
                    )
        if "counter" in self.load_type:
            all_counters = pickle.load(open(os.path.join(self.base_counter_path + results['sample_idx']+'.pkl'), 'rb'))
            for data in all_counters:
                sources.append(
                        [
                            {"from": 'human',
                            "value": f"If you follow the trajectory {data['traj']}, what would happen?"},
                            {"from": 'gpt',
                            "value": ""}
                            ]
                    )

        
        return sources  
    

    def __call__(self, results):
        sources = self.preprocess_vqa(results)
        prompt = f"You are driving a car."


        if self.use_token:
            if not self.desc_qa:
                sources = []
            sources += [[{"from": 'human',
                "value": "Please provide the planning trajectory for the ego car without reasons."},
                {"from": 'gpt',
                "value": "Here is the planning trajectory <waypoint> <waypoint> <waypoint> <waypoint> <waypoint> <waypoint>"}]]
            if self.use_multi_modal_special_token:
                sources[-1][1]['value'] = sources[-1][1]['value'].replace("<waypoint>", self.special_token_list[results['command']])

        if self.use_gen_token:
            if not self.desc_qa:
                sources = []
            sources += [
                [{"from": 'human',
                "value": "Based on the above information, please provide a safe, executable, and reasonable planning trajectory for the ego car."},
                {"from": 'gpt',
                "value": "Here is the planning trajectory <waypoint_ego>"}]
            ]
        if self.planning_qa_command:
            drive_command = results['command']
            sources[-1][0]['value'] = ' ' + self.command_template[drive_command] + ' ' + sources[-1][0]['value']
        # num_new_tokens = self.tokenizer.add_tokens(["<waypoint>"], special_tokens = True)
        vlm_labels = [anno[0]['value'] for anno in sources]

        if self.use_token:
            vqa_anno = [item for pair in sources for item in pair]
            if not self.use_multi_modal_special_token :
                num_new_tokens = self.tokenizer.add_tokens(["<waypoint>"], special_tokens = True)
            else:
                # command "LEFT", "RIGHT", "STRAIGHT", "LANE FOLLOW", "CHANGE LANE LEFT",  "CHANGE LANE RIGHT"
                num_new_tokens = self.tokenizer.add_tokens(["<left_waypoint>"], special_tokens = True)
                num_new_tokens = self.tokenizer.add_tokens(["<right_waypoint>"], special_tokens = True)
                num_new_tokens = self.tokenizer.add_tokens(["<straight_waypoint>"], special_tokens = True)
                num_new_tokens = self.tokenizer.add_tokens(["<follow_waypoint>"], special_tokens = True)
                num_new_tokens = self.tokenizer.add_tokens(["<change_left_waypoint>"], special_tokens = True)
                num_new_tokens = self.tokenizer.add_tokens(["<change_right_waypoint>"], special_tokens = True)
            vqa_anno[0]['value'] = DEFAULT_IMAGE_TOKEN + '\n' + prompt + vqa_anno[0]['value']
            # if self.desc_qa:
            #     vqa_anno[-2]['value'] = DEFAULT_IMAGE_TOKEN + '\n' + prompt + vqa_anno[-2]['value']
        elif self.use_gen_token:
            vqa_anno = [item for pair in sources for item in pair]
            num_new_tokens = self.tokenizer.add_tokens(["<waypoint_ego>"], special_tokens = True)
            vqa_anno[0]['value'] = DEFAULT_IMAGE_TOKEN + '\n' + prompt + vqa_anno[0]['value']
        else:
            vqa_anno = [item for pair in sources for item in pair]
            vqa_anno[0]['value'] = DEFAULT_IMAGE_TOKEN + '\n' + prompt + vqa_anno[0]['value']
            # for anno in sources:
            #     anno[0]['value'] = DEFAULT_IMAGE_TOKEN + '\n' + prompt + anno[0]['value']  
            #     anno[1]['value'] = ''
        
        vqa_converted = preprocess(sources, self.tokenizer, has_image=True, training_mode=False, only_one_system_prompt = True)
        input_ids = vqa_converted['input_ids']

        results['input_ids'] = input_ids
        results['vlm_labels'] = vlm_labels
        
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str



@PIPELINES.register_module()
class LoadAnnoatationMixCriticalVQATest(LoadAnnoatationCriticalVQATest):
    def __init__(
            self,
            with_history_vqa = False,
            single = False,
            conv_template = None,
            is_decoupling=False,
            use_meta_action = False,
            **kwargs
            ):
        super().__init__(**kwargs)
        self.with_history_vqa = with_history_vqa
        self.single = single
        self.conv_template = conv_template
        self.is_decoupling = is_decoupling
        self.use_meta_action = use_meta_action
        if self.conv_template is not None:
            conversation_lib.default_conversation = conversation_lib.conv_templates[self.conv_template]

        if self.conv_template == 'llava_llama_3':
            # TODO:check
            self.tokenizer.eos_token_id = self.tokenizer.convert_tokens_to_ids('<|eot_id|>')
            self.tokenizer.pad_token = self.tokenizer.eos_token
        elif self.conv_template == 'llava_qwen2':
            self.tokenizer.eos_token_id = self.tokenizer.convert_tokens_to_ids('<|im_end|>')
        else:
            self.tokenizer.pad_token = self.tokenizer.unk_token
    
        self.SPEED_SPT = ['<maintain_moderate_speed>','<stop>','<maintain_slow_speed>','<speed_up>','<slow_down>','<maintain_fast_speed>','<slow_down_rapidly>']

    def __call__(self, results):

        sources = self.preprocess_vqa(results)
        prompt = f"You are driving a car."

        if self.with_history_vqa:
  
            scene_change_obj = str(random.choice(critical_obj_sentences))
            sources.append([
                                {"from": 'human',
                                "value": scene_change_obj},
                                {"from": 'gpt',
                                "value": ""}
                                ]
                                )
            light_change = str(random.choice(traffic_light_sentences))
            sources.append([
                                {"from": 'human',
                                "value": light_change},
                                {"from": 'gpt',
                                "value": ""}
                                ]
                    )
            speed_change = "How has the current speed changed compared to the previous frames?"
            sources.append([
                                {"from": 'human',
                                "value": speed_change},
                                {"from": 'gpt',
                                "value": ""}
                                ]
                    )
            behavior_change = "What was my driving behavior in the previous frame?"
            sources.append([
                                {"from": 'human',
                                "value": behavior_change},
                                {"from": 'gpt',
                                "value": ""}
                                ]
                    )
            current_light_qa = str(random.choice(current_traffic_template))
            sources.append([
                                {"from": 'human',
                                "value": current_light_qa},
                                {"from": 'gpt',
                                "value": ""}
                                ]
                    )

            behavior_qa = "What is the current behavior of the vehicle?"
            sources.append([
                                {"from": 'human',
                                "value": behavior_qa},
                                {"from": 'gpt',
                                "value": ""}
                                ]
                    )

        if not self.desc_qa:
                sources = []

        if self.use_meta_action:
            spt = random.choice(self.SPEED_SPT)
            sources += [
                    [{"from": 'human',
                    "value": "What actions should the car be taking?"},
                    {"from": 'gpt',
                        "value": f"The car should be {spt}"}]
                    ]

        if self.use_gen_token:
            
            if self.is_decoupling:
                sources += [
                        [{"from": 'human',
                        "value": "Based on the above information, please provide a safe, executable, and reasonable planning trajectory for the ego car."},
                        {"from": 'gpt',
                        "value": "Here is the planning trajectory <waypoint_ego> <path_waypoint_ego>"}]
                    ]
            else:
                sources += [
                    [{"from": 'human',
                    "value": "Please provide the planning trajectory for the ego car without reasons."},
                    {"from": 'gpt',
                    "value": "Here is the planning trajectory <waypoint_ego>"}]
                ]
            
    
        if self.planning_qa_command:
            drive_command = results['command']
            sources[-1][0]['value'] = ' ' + self.command_template[drive_command] + ' ' + sources[-1][0]['value']
        vlm_labels = [anno[0]['value'] for anno in sources]
        
        if self.use_meta_action:
            new_speed_tokens = ["<maintain_moderate_speed>", "<stop>", "<maintain_slow_speed>", "<speed_up>",'<slow_down>','<maintain_fast_speed>','<slow_down_rapidly>']
            new_path_tokens = ["<lanefollow>", '<straight>', '<turn_left>','<change_lane_left>','<turn_right>','<change_lane_right>']
            num_new_tokens = self.tokenizer.add_tokens(new_speed_tokens, special_tokens = True)            
            num_new_tokens = self.tokenizer.add_tokens(new_path_tokens, special_tokens = True)   

        if self.use_gen_token:
            vqa_anno = [item for pair in sources for item in pair]
            num_new_tokens = self.tokenizer.add_tokens(["<waypoint_ego>"], special_tokens = True)
            if self.is_decoupling:
                num_new_tokens = self.tokenizer.add_tokens(["<path_waypoint_ego>"], special_tokens = True)

            if self.single:
                for i in range(len(vqa_anno)):
                    if i % 2 == 0:  # 检查i是否为偶数
                        vqa_anno[i]['value'] = DEFAULT_IMAGE_TOKEN + '\n' + prompt + vqa_anno[i]['value']
            else:
                vqa_anno[0]['value'] = DEFAULT_IMAGE_TOKEN + '\n' + prompt + vqa_anno[0]['value']
        else:
            for anno in sources:
                anno[0]['value'] = DEFAULT_IMAGE_TOKEN + '\n' + prompt + anno[0]['value']  
                anno[1]['value'] = ''
            vqa_anno = [item for pair in sources for item in pair]
        
        if self.conv_template == 'llava_llama_3':
            vqa_converted = preprocess_llama3([vqa_anno], self.tokenizer, True,  training_mode=False, only_one_system_prompt = False)
        elif self.conv_template == 'llava_qwen2':
            vqa_converted = preprocess_qwen2([vqa_anno], self.tokenizer, True, training_mode=False, only_one_system_prompt = False)
        else:
            vqa_converted = preprocess(sources, self.tokenizer, has_image=True, training_mode=False, only_one_system_prompt = False)

        
        input_ids = vqa_converted['input_ids']

        results['input_ids'] = input_ids
        results['vlm_labels'] = vlm_labels
        
        return results