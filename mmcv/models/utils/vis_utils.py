import numpy as np
import torch
import torch.nn.functional as F
import cv2
import os
import matplotlib.cm as cm
from PIL import Image, ImageDraw, ImageFont
from mmcv.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes
from mmcv.parallel import DataContainer as DC

# 基础颜色定义，使用BGR编码

_GREEN2 = (0, 204, 0)
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)
_RED = (0, 0, 255)  # 红色
_GREEN = (0, 255, 0)  # 绿色
_BLUE = (255, 51, 51)  # 蓝色
_BLUE2 = (158, 168, 3)
_YELLOW = (0, 255, 255)  # 黄色
_ORANGE = (0, 125, 255)  # 橙色
_GREEN3 = (0, 100, 50)
_PURPLE = (128, 0, 128)  # 紫色
_PINK = (180, 105, 255)  # 粉红色
_GRAY = (128, 128, 128)  # 灰色
_static = (229, 35, 98)
_ORANGE2 = (255, 120, 153)  # 淡橙色
_BLUE3 = (120, 120, 255)  # 浅蓝色

nuscenes_set_colors = {
    ("car",): _BLUE,
    ("truck", "construction_vehicle", "bus", "trailer"): _BLUE2,
    ("motorcycle", "bicycle"): _GREEN,
    ("pedestrian",): _YELLOW,
    ("traffic_cone",): _ORANGE
}
nuscenes_set_colors = {cls: color for clses, color in nuscenes_set_colors.items() for cls in clses}

bhx_set_colors = {
    ('car', 'sport_utility_vehicle', 'van', 'Car', 'dummy_car', 'unknown'): _BLUE,
    ('bus', 'tanker', 'truck', 'trailer', 'other_vehicle', 'Truck',): _BLUE2,
    ('cyclist', 'motorcyclist', 'tricyclist', 'parked_cycle', 'Cyclist', "handcart", 'dummy_cyclist',
     'bicycle', 'motorcycle',): _GREEN,
    ('pedestrian', 'Pedestrian', 'people', 'dummy',): _YELLOW,
    ("traffic_cone", "barrier", "crash_pile",): _ORANGE
}
bhx_set_colors = {cls: color for clses, color in bhx_set_colors.items() for cls in clses}

# 目标框颜色设置
b2d_set_colors = {
    ('car', 'van',): _GREEN3,
    ('truck',): _GREEN3,
    ('bicycle',): _GREEN2,
    ('pedestrian',): _GREEN2,
    ("traffic_sign", "traffic_cone", "traffic_light",): _RED,
    ('others',): _PURPLE,
}

b2d_set_colors = {cls: color for clses, color in b2d_set_colors.items() for cls in clses}

trk_colors = np.random.randint(256, size=(30, 3)).tolist()

merge_h = 600  # 2 x 3
merge_w = 900

# 车道线颜色设置
ld_set_colors = {
    "Broken": _GRAY,
    "Solid": _BLACK,
    "SolidSolid": _BLACK,
    "Center": _GRAY,  # 白色
    "TrafficLight": _RED,
    "StopSign": _RED,
    "waypoint": _BLUE,
    "waypointGT": _GREEN,
    "waypointInactive": _ORANGE,
    "pw_waypoint": _ORANGE2,
    "pwwaypointInactive": _BLUE3,
}

# # 使用渐变色
# ld_set_colors["Broken"] = gradient_color(_GRAY, _BLUE, 10)[-1]
# ld_set_colors["waypointInactive"] = gradient_color(_ORANGE, _YELLOW, 10)[-1]

def vis_depth_on_img(img, depth_map, save_path, max_depth=80):
    depth_map[depth_map > 80] = 80
    color_map = cm.get_cmap(name='jet')
    h, w = depth_map.shape
    for col in range(h):
        for row in range(w):
            if depth_map[col, row] != 0:
                pt_color_value = min(255, int(min(depth_map[col, row], max_depth) / max_depth * 255))
                color = color_map(pt_color_value)
                color = [c * 255 for c in color[:3]]
                cv2.circle(img, (row, col), 1, color)
    cv2.imwrite(save_path, img)


def get_rotz_matrix_array(rz):
    """
    :param rz: [N, ]
    :return: [N, 3, 3]
    """
    bbox_num = rz.shape[0]
    temp_zeros = np.zeros_like(rz)
    temp_ones = np.ones_like(rz)
    mat = [np.cos(rz), -np.sin(rz), temp_zeros.copy(),
           np.sin(rz), np.cos(rz), temp_zeros.copy(),
           temp_zeros.copy(), temp_zeros.copy(), temp_ones.copy()]
    return np.stack(mat, axis=1).reshape(bbox_num, 3, 3)


def gen_dx_bx(xbound, ybound, zbound, **kwargs):
    dx = np.array([row[2] for row in [xbound, ybound, zbound]])
    bx = np.array([row[0] for row in [xbound, ybound, zbound]])
    nx = np.array([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]], dtype=np.long)

    return dx, bx, nx

def gen_3d_object_corners_array(location, dimension, yaw):
    """
    :param location: [N, 3]
    :param dimension: [N, 3]
    :param yaw: [N,]
    :return: [N, 8, 3]
    """
    l, w, h = dimension[:, 0:1], dimension[:, 1:2], dimension[:, 2:3]
    x_corners = np.concatenate([l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2], axis=1)
    y_corners = np.concatenate([w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], axis=1)
    z_corners = np.concatenate([-h / 2., -h / 2., -h / 2., -h / 2., h / 2., h / 2., h / 2., h / 2.], axis=1)
    box_points_coords = np.stack((x_corners, y_corners, z_corners), axis=1)

    mat = get_rotz_matrix_array(yaw)
    corners_3d = np.matmul(mat, box_points_coords)
    corners_3d = corners_3d + location[:, :, np.newaxis]
    corners_3d = corners_3d.transpose(0, 2, 1)
    return corners_3d


def get_rotz_matrix(rz):
    # rz: float
    mat = [np.cos(rz), -np.sin(rz), 0,
           np.sin(rz), np.cos(rz), 0,
           0, 0, 1]
    return mat


def get_rotz_matrix_tensor(rz):
    """
    :param rz: [N, ]
    :return: [N, 3, 3]
    """
    bbox_num = rz.shape[0]
    temp_zeros = torch.zeros_like(rz)
    temp_ones = torch.ones_like(rz)
    mat = [torch.cos(rz), -torch.sin(rz), temp_zeros.clone(),
           torch.sin(rz), torch.cos(rz), temp_zeros.clone(),
           temp_zeros.clone(), temp_zeros.clone(), temp_ones.clone()]
    return torch.stack(mat, dim=1).reshape(bbox_num, 3, 3)


def gen_3d_object_corners_tensor(location, dimension, yaw):
    """
    :param location: [N, 3]
    :param dimension: [N, 3]
    :param yaw: [N,]
    :return: [N, 8, 3]
    """
    l, w, h = dimension[:, 0:1], dimension[:, 1:2], dimension[:, 2:3]
    x_corners = torch.cat([l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2], dim=1)
    y_corners = torch.cat([w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], dim=1)
    z_corners = torch.cat([-h / 2., -h / 2., -h / 2., -h / 2., h / 2., h / 2., h / 2., h / 2.], dim=1)
    box_points_coords = torch.stack((x_corners, y_corners, z_corners), dim=1)

    mat = get_rotz_matrix_tensor(yaw)
    corners_3d = torch.matmul(mat, box_points_coords)

    corners_3d = corners_3d + location.unsqueeze(-1)
    corners_3d = corners_3d.permute(0, 2, 1)
    return corners_3d


def gen_3d_object_corners(location, dimension, yaw):
    """
    gen 3d 8 corners
    :param location: xyz, [3,]
    :param dimension: lwh, [3,]
    :param yaw: int
    :return: [8, 3]
    """

    l, w, h = dimension[0], dimension[1], dimension[2]
    x_corners = np.array([l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2])
    y_corners = np.array([w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2])
    z_corners = np.array([-h / 2., -h / 2., -h / 2., -h / 2., h / 2., h / 2., h / 2., h / 2.])
    box_points_coords = np.vstack((x_corners, y_corners, z_corners))

    mat = get_rotz_matrix(yaw)
    mat = np.array(mat).reshape(3, 3)

    corners_3d = np.matmul(mat, box_points_coords)
    corners_3d = corners_3d + np.array(location[0:3]).reshape(3, 1)
    corners_3d = corners_3d.transpose()
    return corners_3d


def lidarcorners_to_2dboxs_tensor(corners_3d_lidar, lidar2cam, intrin, remap_shape):
    """
    :param corners_3d_lidar: [N, 8, 4]
    :param lidar2cam: [4, 4]
    :param intrin: [3, 3]
    :param remap_shape: [w, h]
    :return:
    """
    if intrin.shape[0] == 4:
        intrin = intrin[:3, :3]

    img_w, img_h = remap_shape

    corners_3d_cam = lidar2cam[None, ...] @ corners_3d_lidar.permute(0, 2, 1)
    corners_2d = intrin[None, ...] @ corners_3d_cam[:, :3, :]
    corners_2d = corners_2d.permute(0, 2, 1)
    corners_2d[:, :, :2] = corners_2d[:, :, :2] / corners_2d[:, :, 2:3]

    inds = (corners_2d[:, :, 0] > 0) & (corners_2d[:, :, 0] < img_w) & \
           (corners_2d[:, :, 1] > 0) & (corners_2d[:, :, 1] < img_h) & \
           (corners_2d[:, :, 2] > 0)
    valid_mask = inds.sum(1) >= 2
    corners_2d = corners_2d[:, :, :2]  # [N, 8, 2]

    assert corners_3d_lidar.shape[0] == corners_2d.shape[0]
    bbox_num = corners_3d_lidar.shape[0]

    bbox_2ds = []
    for bbox_i in range(bbox_num):
        corner_2d = corners_2d[bbox_i, ...]
        x_min = max(0, corner_2d[:, 0].min())
        x_max = min(img_w, corner_2d[:, 0].max())
        y_min = max(0, corner_2d[:, 1].min())
        y_max = min(img_h, corner_2d[:, 1].max())
        bbox_2ds.append([x_min, y_min, x_max, y_max])
    bbox_2ds = torch.tensor(bbox_2ds, dtype=torch.float, device=corners_3d_lidar.device)
    return bbox_2ds, valid_mask


def lidarcorners_to_2dboxs(corners_3d_lidar, lidar2cam, intrin, remap_shape):
    """
    :param corners_3d_lidar: [N, 8, 4]
    :param lidar2cam: [4, 4]
    :param intrin: [4, 4]
    :param remap_shape: [w, h]
    :return:
    """
    img_w, img_h = remap_shape
    intrin = intrin[:3, :3]

    corners_3d_cam = np.matmul(lidar2cam[np.newaxis, ...],
                               corners_3d_lidar.transpose((0, 2, 1)))

    corners_2d = np.matmul(intrin[np.newaxis, ...],
                           corners_3d_cam[:, :3, :]).transpose((0, 2, 1))
    corners_2d[:, :, :2] = corners_2d[:, :, :2] / corners_2d[:, :, 2:3]

    inds = (corners_2d[:, :, 0] > 0) & (corners_2d[:, :, 0] < img_w) & \
           (corners_2d[:, :, 1] > 0) & (corners_2d[:, :, 1] < img_h) & \
           (corners_2d[:, :, 2] > 0)
    valid_mask = inds.sum(1) >= 2

    # corners_2d = corners_2d[valid_mask, ...]
    corners_2d = corners_2d[:, :, :2]  # [N, 8, 2]

    assert corners_3d_lidar.shape[0] == corners_2d.shape[0]
    bbox_num = corners_3d_lidar.shape[0]

    bbox_2ds = []
    for bbox_i in range(bbox_num):
        corner_2d = corners_2d[bbox_i, ...]
        x_min = max(0, corner_2d[:, 0].min())
        x_max = min(img_w, corner_2d[:, 0].max())
        y_min = max(0, corner_2d[:, 1].min())
        y_max = min(img_h, corner_2d[:, 1].max())
        bbox_2ds.append([x_min, y_min, x_max, y_max])
    return bbox_2ds, valid_mask


def draw_box_3d(image, corners, pred_dict,tasks, c=(0, 0, 255), direction_c=_ORANGE, thickness=2, cam=None):
    """
    :param image:  [256, 512, 3]
    :param corners: [8, 2], 2d corners
    :param c: colors
    :return:
    """
    extra_txt = ""
    if 'cipo' in pred_dict and pred_dict['cipo']!=-1:
        extra_txt += f"{str(pred_dict['cipo']+1)} "
    if 'subclass' in pred_dict:
        subclass_names = [k['names'] for k in tasks if k['task_name'] == 'subclass'][0]
        extra_txt += f"{subclass_names[pred_dict['subclass']]} "
    else:
        subclass_names = [k['names'] for k in tasks if k['task_name'] == 'class'][0]
        extra_txt += f"{subclass_names[pred_dict['class']]} "
    if cam in ['mid_center_top_wide','mid_center_top_tele','crop_f30_2']:
        if pred_dict.get('brake_light',0) == 1:
            c = (0,0,255)
        if pred_dict.get('high_brake_light',0) == 1:
            c = (0,0,255)
            extra_txt += "H "
        if pred_dict.get('side_brake_light',0) == 1:
            c = (0,0,255)
            extra_txt += f"S "
    if pred_dict.get('cross_lane',0) == 1:
        c  = (255,255,255)
    if extra_txt:
        cv2.putText(image,extra_txt.replace("sport_utility_vehicle","SUV"),(int(min(corners[:,0])),int(min(corners[:,1]-10))),cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,0,255),2)

    face_idx = [[0, 1, 5, 4],
                [1, 2, 6, 5],
                [2, 3, 7, 6],
                [3, 0, 4, 7]]
    im_h, im_w = image.shape[:2]
    valid = (corners[:, 0] > 0) * (corners[:, 0] < im_w) * \
            (corners[:, 1] > 0) * (corners[:, 1] < im_h)
    for ind_f in range(3, -1, -1):
        f = face_idx[ind_f]
        for j in range(4):
            if valid[f[j]] and valid[f[(j + 1) % 4]]:
                cv2.line(image, (int(corners[f[j], 0]), int(corners[f[j], 1])),
                         (int(corners[f[(j + 1) % 4], 0]), int(corners[f[(j + 1) % 4], 1])), c, thickness, lineType=cv2.LINE_AA)
        if ind_f == 0:
            if valid[f[0]] and valid[f[2]]:
                cv2.line(image, (int(corners[f[0], 0]), int(corners[f[0], 1])),
                         (int(corners[f[2], 0]), int(corners[f[2], 1])), direction_c, thickness, lineType=cv2.LINE_AA)
            if valid[f[1]] and valid[f[3]]:
                cv2.line(image, (int(corners[f[1], 0]), int(corners[f[1], 1])),
                         (int(corners[f[3], 0]), int(corners[f[3], 1])), direction_c, thickness, lineType=cv2.LINE_AA)

    return image



def gen_bev_painter(bev_painter_h, bev_painter_w, grid_config, ratio, background=220):
    img_bev = np.full((bev_painter_h, bev_painter_w, 3), background, dtype=np.uint8)

    x_min, x_max, x_interval = grid_config["xbound"]
    y_min, y_max, y_interval = grid_config["ybound"]

    for x in range(int(x_min) // 10 * 10, int(x_max) // 10 * 10 + 1, 10):
        canvas_bev_x = int((x - x_min) / x_interval * ratio)
        canvas_bev_y = int(bev_painter_h)
        cv2.line(img_bev, (canvas_bev_x, canvas_bev_y),
                 (canvas_bev_x, canvas_bev_y - 10), _GRAY, 2)
        cv2.putText(img_bev, str(x) + "m", (canvas_bev_x - 10, canvas_bev_y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, _GRAY, 1)

    for y in range(int(y_min) // 10 * 10, int(y_max) // 10 * 10 + 1, 10):
        canvas_bev_x = bev_painter_w
        canvas_bev_y = bev_painter_h - int((y - y_min) / y_interval * ratio)
        cv2.line(img_bev, (canvas_bev_x, canvas_bev_y),
                 (canvas_bev_x - 10, canvas_bev_y), _GRAY, 2)
        cv2.putText(img_bev, str(y) + "m", (canvas_bev_x - 40, canvas_bev_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, _GRAY, 1)
    return img_bev


def draw_bev_bboxes(grid_config, corners_3d_lidar, tasks,
                    pred_dict, set_colors, filter_cids, vis_dir_cids,
                    rot_uncerts=None, bev_h=None, trk_check=False,
                    show_extra_txt=None):
    """
    :param grid_config:
    :param corners_3d_lidar: [N, 8, 3]
    :return:
    """
    
    dx, bx, nx = gen_dx_bx(**grid_config)

    if bev_h is None:
        bev_h = merge_h

    ratio = bev_h / int(nx[1])
    bev_painter_h = bev_h

    bev_painter_w = int(nx[0] * ratio)
    # img_bev = np.full((bev_painter_h, bev_painter_w, 3), 128, dtype=np.uint8)
    img_bev = gen_bev_painter(bev_painter_h, bev_painter_w, grid_config, ratio)

    # draw selfcar
    selfcar_position = np.array([[-0.75, 1.5], [0.75, -1.5]])
    selfcar_bev_x = ((selfcar_position[:, 0] - bx[0]) / dx[0] * ratio).astype(np.int)
    selfcar_bev_y = bev_painter_h - ((selfcar_position[:, 1] - bx[1]) / dx[1] * ratio).astype(np.int)
    cv2.rectangle(img_bev, (selfcar_bev_x[0], selfcar_bev_y[0]),
                  (selfcar_bev_x[1], selfcar_bev_y[1]), _GRAY, -1)

    if rot_uncerts is not None:
        assert rot_uncerts.shape[0] == corners_3d_lidar.shape[0]

    # draw bev
    corners_bev_lidar = corners_3d_lidar[:, :4, :2]
    for bbox_i in range(corners_bev_lidar.shape[0]):
        pts_bev = corners_bev_lidar[bbox_i, ...]
        pts_bev[:, 0] = (pts_bev[:, 0] - bx[0]) / dx[0]
        pts_bev[:, 1] = (pts_bev[:, 1] - bx[1]) / dx[1]

        pts_center = pts_bev.mean(0)
        velo = pred_dict['box'][bbox_i, 7:9]
        next_pts_center = pts_center + velo

        pts_bev = (pts_bev * ratio).astype(np.int)
        pts_bev[:, 1] = bev_painter_h - pts_bev[:, 1]

        pts_center = (pts_center * ratio).astype(np.int)
        pts_center[1] = bev_painter_h - pts_center[1]

        next_pts_center = (next_pts_center * ratio).astype(np.int)
        next_pts_center[1] = bev_painter_h - next_pts_center[1]

        box_task_names = [k['task_name'] for k in tasks if k['level'] == 'box']
        cur_pred_dict = {}

        for t in box_task_names:
            cur_pred_dict[t] = pred_dict[t][bbox_i].copy()
        # label = labels_3d[bbox_i]
        # static = statics_3d[bbox_i]
        if cur_pred_dict['class'] in filter_cids:
            continue
        class_name = tasks[0]['names'][cur_pred_dict['class']]
        assert class_name in set_colors
        
        bboxes_3d = pred_dict['box']
        if 'score' in pred_dict:
            scores_3d = pred_dict['score']
        else:
            scores_3d = None
        # if bboxes_3d.shape[1] == 10:
        #     color = trk_colors[int(bboxes_3d[bbox_i, 9]) % 30]  if cur_pred_dict.get('static',1) else (0,0,0)
        # else:
        #     color = set_colors[class_name] if cur_pred_dict.get('static',1) else (0,0,0)

        color = set_colors[class_name] if cur_pred_dict.get('static', 1) else (0,0,0)
        cv2.polylines(img_bev, [pts_bev], True, color, 2)

        if cur_pred_dict['class'] in vis_dir_cids:
            cv2.line(img_bev, pts_bev[0], pts_bev[1], color=_ORANGE, thickness=2)

        # draw vxvy

        # speed = bboxes_3d[bbox_i, 7:9]
        # speed_norm = np.sqrt(speed[0] * speed[0] + speed[1] * speed[1])
        # if speed_norm < 0.3:
        #     cv2.polylines(img_bev, [pts_bev], True, _static, 2)

        # cv2.arrowedLine(img_bev, pts_center, next_pts_center,
        #                 color=_RED, thickness=2)

        # eval_online show(velo > 500 must crop_model pred bbox)
        velo = bboxes_3d[bbox_i, 7:9]
        if velo.max() > 800:
            cv2.polylines(img_bev, [pts_bev], True, _PURPLE, 2)

        # temp zoom velo vis
        temp_velo_zoom_ratio = 2
        draw_next_pts_center = (next_pts_center + pts_center) // temp_velo_zoom_ratio
        if velo.max() > 800:
            draw_next_pts_center = pts_center       # eval_online_show bbox
        cv2.arrowedLine(img_bev, pts_center, draw_next_pts_center,
                        color=_ORANGE, thickness=2)

        # draw uncert
        if rot_uncerts is not None:
            rot_uncert = rot_uncerts[bbox_i]
            yaw = bboxes_3d[bbox_i, 6]
            if yaw < 0:
                yaw += 2 * np.pi
            # img_bev = cv2.circle(img_bev, pts_center, int(rot_uncert * 10), (0, 0, 0), 1)
            yaw_angle = -yaw * 180 / np.pi + 90
            cv2.ellipse(img_bev, pts_center, (int(24 * rot_uncert), 12),
                        angle=yaw_angle, startAngle=0, endAngle=360, color=(0, 0, 0), thickness=1, lineType=1)

        # if scores_3d is not None:
        #     cv2.putText(img_bev, str("%.2f"%(scores_3d[bbox_i])),
        #                 (int(pts_bev[0][0]), int(pts_bev[0][1])),
        #                 fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        #                 fontScale=0.5,
        #                 color=color,
        #                 thickness=1)
    if 'scene' in pred_dict:
        scene_map = {0: 'Pilot', 1: 'Parking'}
        cv2.putText(img_bev, scene_map[int(pred_dict['scene'])],(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),1)

    if show_extra_txt is not None:
        cv2.putText(img_bev, str(show_extra_txt), (bev_painter_w//2-10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    return img_bev, ratio


def get_lines_points_cams(road_edge_points, dx, bx, cam2lidar, intrins, upsample=0.2):
    road_edge_points = np.array(road_edge_points, dtype=np.float32).copy()

    road_edge_points[:, 0] = road_edge_points[:, 0] * dx[0] * upsample + bx[0]
    road_edge_points[:, 1] = road_edge_points[:, 1] * dx[1] * upsample + bx[1]

    road_edge_points = np.concatenate(
        [road_edge_points, np.zeros((road_edge_points.shape[0], 1))], axis=1
    )
    road_edge_points = np.concatenate(
        [road_edge_points, np.ones((road_edge_points.shape[0], 1))], axis=1
    )
    num_cams = cam2lidar.shape[0]
    road_edge_points = np.tile(road_edge_points[np.newaxis], (num_cams, 1, 1))
    cam_roadseg_points = np.linalg.inv(cam2lidar) @ road_edge_points.transpose((0, 2, 1))
    ims_roadseg_points = intrins @ cam_roadseg_points[:, :3, :]
    ims_roadseg_points = ims_roadseg_points.transpose((0, 2, 1))  # [6, N, 3]
    ims_roadseg_points[:, :, :2] = ims_roadseg_points[:, :, :2] / ims_roadseg_points[:, :, 2:3]
    return ims_roadseg_points


def draw_bbox_imgs(imgs,
                   intrins,
                   rots,
                   trans,
                   cams,
                   pred_dict,
                   tasks,
                   corners_3d_lidar,
                   filter_cids,
                   set_colors,
                   post_rots=None,
                   post_trans=None):
    draw_imgs = []

    for i, cam in enumerate(cams):

        img = imgs[i]
        img_h, img_w = img.shape[:2]
        intrin = intrins[i]
        rot = rots[i]
        tran = trans[i]

        cam2lidar = np.eye(4)
        cam2lidar[:3, :3] = rot
        cam2lidar[:3, 3] = tran
        lidar2cam = np.linalg.inv(cam2lidar)
        corners_3d_cam = np.matmul(lidar2cam[np.newaxis, ...],
                                   corners_3d_lidar.transpose((0, 2, 1)))

        corners_2d = np.matmul(intrin[np.newaxis, ...],
                               corners_3d_cam[:, :3, :]).transpose((0, 2, 1))

        corners_2d[:, :, :2] = corners_2d[:, :, :2] / corners_2d[:, :, 2:3]

        if post_rots is not None and post_trans is not None:
            post_rot = post_rots[i][:2, :2][None, ...]  # [1, 2, 2]
            post_tran = post_trans[i][:2][None, ...]  # [1, 2]

            corners_2d_xy = corners_2d[:, :, :2]
            corners_2d_z = corners_2d[:, :, 2:3]
            corners_2d_xy = np.matmul(post_rot, np.transpose(corners_2d_xy, (0, 2, 1))).transpose((0, 2, 1))
            corners_2d_xy = corners_2d_xy + post_tran[:, None, :]  # [N, 8, 2]
            corners_2d = np.concatenate([corners_2d_xy, corners_2d_z], axis=-1)

        inds = (corners_2d[:, :, 0] > 0) & (corners_2d[:, :, 0] < img_w) & \
               (corners_2d[:, :, 1] > 0) & (corners_2d[:, :, 1] < img_h) & \
               (corners_2d[:, :, 2] > 0)
        valid_mask = inds.sum(1) >= 4

        corners_2d = corners_2d[valid_mask, ...]
        corners_2d = corners_2d[:, :, :2]

        box_task_names = [k['task_name'] for k in tasks if k['level'] == 'box']

        cur_pred_dict = pred_dict.copy()
        for t in box_task_names:
            cur_pred_dict[t] = cur_pred_dict[t][valid_mask]

        for bbox_i in range(corners_2d.shape[0]):
            # score = scores[bbox_i]
            tmp_pred_dict = cur_pred_dict.copy()
            for t in box_task_names:
                tmp_pred_dict[t] = cur_pred_dict[t][bbox_i]

            if tmp_pred_dict['class'] in filter_cids:
                continue
            # if tmp_pred_dict['box'].shape[1] == 10:
            #     color = trk_colors[int(tmp_pred_dict['box'][valid_mask][bbox_i, 9]) % 30]
            # else:
            #     color = set_colors[tasks[0]['names'][tmp_pred_dict['class']]]

            color = set_colors[tasks[0]['names'][tmp_pred_dict['class']]]

            # NOTE@Jianfeng: pass
            # draw_box_3d(img, corners_2d[bbox_i], tmp_pred_dict, tasks, c=color, cam=cam)
        draw_imgs.append(img)

    img_to_show = multicam_show_imgs(cams, draw_imgs)
    return img_to_show


def show_multicam_bboxes(imgs, intrins, rots, trans, cams,
                         grid_config, timestamp, tasks, pred_dict,
                         filter_cids=[5], dataset_type='nuscenes', rot_uncerts=None,
                         trk_check=False, bev_paint_extra_info=None,
                         draw_bbox_img=True, draw_aug_img=False,
                         aug_imgs=None, post_rots=None, post_trans=None):
    """
    :param imgs:  [len(cams), H, W]
    :param intrins:  [len(cams), 3, 3]
    :param rots:  [len(cams), 3, 3]
    :param trans:  [len(cams), 3]
    :param bboxes_3d:  [N, 9](without tracking) or [N, 10](with tracking)
    :param scores_3d: [N, ]
    :param labels_3d: [N, ]
    :param cams:
    :param grid_config: {'xbound': [-51.2, 51.2, 0.8],
                         'ybound': [-51.2, 51.2, 0.8],
                         'zbound': [-10.0, 10.0, 20.0],
                         'dbound': [1.0, 60.0, 1.0],}
    :param timestamp:
    :return:
    """
    if dataset_type == 'nuscenes':
        set_colors = nuscenes_set_colors
        vis_dir_cids = [0, 1, 2, 3, 4, 6, 7]
    elif dataset_type == 'bhx':
        set_colors = bhx_set_colors
        vis_dir_cids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12]
    elif dataset_type == 'b2d':
        set_colors = b2d_set_colors
        vis_dir_cids = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    else:
        raise NotImplementedError()

    corners_3d_lidar = gen_3d_object_corners_array(location=pred_dict['box'][:, :3],
                                                   dimension=pred_dict['box'][:, 3:6],
                                                   yaw=pred_dict['box'][:, 6])
    corners_3d_lidar = np.concatenate([corners_3d_lidar,
                                       np.ones((corners_3d_lidar.shape[0], 8, 1))], axis=2)

    if len(cams) > 6:
        bev_h = merge_h // 2 * 4
    else:
        bev_h = merge_h

    img_bev, ratio = draw_bev_bboxes(grid_config,
                              corners_3d_lidar.copy(),
                              tasks,
                              pred_dict,
                              filter_cids=filter_cids,
                              set_colors=set_colors,
                              vis_dir_cids=vis_dir_cids,
                              rot_uncerts=rot_uncerts,
                              bev_h=bev_h,
                              trk_check=trk_check,
                              show_extra_txt=bev_paint_extra_info)

    img_to_show = None
    if draw_bbox_img:
        img_to_show = draw_bbox_imgs(imgs=imgs,
                                     intrins=intrins,
                                     rots=rots,
                                     trans=trans,
                                     cams=cams,
                                     pred_dict=pred_dict,
                                     tasks=tasks,
                                     corners_3d_lidar=corners_3d_lidar,
                                     filter_cids=filter_cids,
                                     set_colors=set_colors)

    if draw_aug_img:
        assert aug_imgs is not None
        aug_img_to_show = draw_bbox_imgs(imgs=aug_imgs,
                                         intrins=intrins,
                                         rots=rots,
                                         trans=trans,
                                         cams=cams,
                                         pred_dict=pred_dict,
                                         tasks=tasks,
                                         corners_3d_lidar=corners_3d_lidar,
                                         filter_cids=filter_cids,
                                         set_colors=set_colors,
                                         post_rots=post_rots,
                                         post_trans=post_trans)

        img_to_show = np.concatenate([aug_img_to_show, img_to_show], axis=1)

    # return all_img_show
    return img_to_show, img_bev, ratio


def draw_bev_points(img_bev, grid_config, roadseg_points):
    """
    :param merge_h:
    :param merge_w:
    :param grid_config:
    :param roadseg_points: [N, 3]
    :return:
    """
    
    dx, bx, nx = gen_dx_bx(**grid_config)
    ratio = merge_h / int(nx[1])
    bev_painter_h, bev_painter_w = img_bev.shape[:2]

    # bev_painter_h = merge_h
    # bev_painter_w = int(nx[0] * ratio)
    # img_bev = gen_bev_painter(bev_painter_h, bev_painter_w, dx, bx, nx, ratio)

    roadseg_points[:, 0] = (roadseg_points[:, 0] - bx[0]) / dx[0] * ratio
    roadseg_points[:, 1] = bev_painter_h - (roadseg_points[:, 1] - bx[1]) / dx[1] * ratio

    roadseg_points = roadseg_points.astype(int)

    valid_mask = (roadseg_points[:, 0] > 0) & (roadseg_points[:, 0] < bev_painter_w) & \
                 (roadseg_points[:, 1] > 0) & (roadseg_points[:, 1] < bev_painter_h)
    roadseg_points = roadseg_points[valid_mask]

    img_bev[roadseg_points[:, 1], roadseg_points[:, 0]] = _YELLOW
    return img_bev

def format_bbox(bbox):
    if isinstance(bbox, LiDARInstance3DBoxes):
        bbox = format_bbox(bbox.tensor)
    elif isinstance(bbox, DC):
        bbox = format_bbox(bbox._data)
    elif isinstance(bbox, torch.Tensor):
        bbox = bbox.detach().cpu().numpy()
    return bbox

def multicam_show_imgs(cams, draw_imgs):
    resize_h = merge_h // 2
    resize_w = merge_w // 3
    img_to_show = np.zeros((merge_h, merge_w, 3))

    if len(cams) > 1 and len(cams) <= 6:
        for cam_i, cam in enumerate(cams[:6]):
            col_idx = cam_i // 3
            row_idx = cam_i % 3
            img = cv2.resize(draw_imgs[cam_i].copy(), (resize_w, resize_h))
            img_to_show[int(col_idx * resize_h): int((col_idx + 1) * resize_h),
            int(row_idx * resize_w): int((row_idx + 1) * resize_w), :] = img
    elif len(cams) > 6:
        for cam_i, cam in enumerate(cams[:6]):
            col_idx = cam_i // 3
            row_idx = cam_i % 3
            img = cv2.resize(draw_imgs[cam_i].copy(), (resize_w, resize_h))
            img_to_show[int(col_idx * resize_h): int((col_idx + 1) * resize_h),
            int(row_idx * resize_w): int((row_idx + 1) * resize_w), :] = img

        cropf30_extra_cams_paints = np.zeros((resize_h, merge_w, 3))
        cropr60_extra_cams_paints = np.zeros((resize_h, merge_w, 3))
        for cam_i, cam in enumerate(cams[6:]):      # temp only f30
            img = cv2.resize(draw_imgs[cam_i + 6].copy(), (resize_w, resize_h))
            if cam == "mid_center_top_tele" or cam == "crop_f30_2":
                cropf30_extra_cams_paints[0: int(resize_h), int(resize_w): int(2 * resize_w)] = img
            elif cam == "crop_r60":
                cropr60_extra_cams_paints[0: int(resize_h), int(resize_w): int(2 * resize_w)] = img

        img_to_show = np.concatenate([cropf30_extra_cams_paints, img_to_show, cropr60_extra_cams_paints], axis=0)

    elif len(cams) == 1:
        img_to_show = cv2.resize(draw_imgs[0].copy(), (merge_w, merge_h))
    return img_to_show

def draw_img_ld(imgs, calib_infos, cams, map_res, is_aug=False, draw_point=False):
    """
    :param imgs:
    :param intrins: [cam_num, 3, 3]
    :param rots: [cam_num, 3, 3]
    :param trans: [cam_num, 3]
    """
    intrins = calib_infos["intrins"]
    rots = calib_infos["rots"]
    trans = calib_infos["trans"]
    # post_rots = calib_infos["post_rots"]
    # post_trans = calib_infos["post_trans"]

    lane_pts = map_res["lane_pts_3d"].copy()  # [lane_num, num_pts, 2]
    # lane_labels = map_res["lane_labels_3d"].copy()
    lane_colors = map_res["lane_colors"]
    lane_sizes = map_res["lane_sizes"]
    lane_grad_color = map_res["lane_grad_color"]
    lane_ignore = map_res["lane_ignore_img"]

    # Sample more points
    if len(lane_pts) != 0:
        # lane_pts = np.stack(
        #     [
        #         interp_utils.interp_arc(t=500, points=lane_pts[lane_i])
        #         for lane_i in range(len(lane_pts))
        #     ]
        # )
        # fake z
        if lane_pts.shape[-1] == 2:
            lane_pts = np.concatenate(
                (
                    lane_pts,
                    np.zeros_like(lane_pts[:, :, 0:1]),
                ),
                axis=-1,
            )
        lane_pts = np.concatenate(
            (
                lane_pts,
                np.ones_like(lane_pts[:, :, 0:1]),
            ),
            axis=-1,
        )

    draw_imgs = []
    for cam_i, cam in enumerate(cams):
        img = imgs[cam_i]
        img_h, img_w = img.shape[:2]
        intrin = intrins[cam_i]
        rot = rots[cam_i]
        tran = trans[cam_i]

        cam2lidar = np.eye(4)
        cam2lidar[:3, :3] = rot
        cam2lidar[:3, 3] = tran
        lidar2cam = np.linalg.inv(cam2lidar)

        if len(lane_pts) != 0:
            # FIXME: astype(np.float32) fix the wrong result, do not know why?
            lane_pts_cam = np.matmul(
                lidar2cam[np.newaxis, ...],
                lane_pts.transpose(0, 2, 1).astype(np.float32),
            )
            lane_pts_2d = np.matmul(
                intrin[np.newaxis, ...], lane_pts_cam[:, :3, :]
            ).transpose((0, 2, 1))
            lane_pts_2d[:, :, :2] = lane_pts_2d[:, :, :2] / lane_pts_2d[:, :, 2:3]

            # if is_aug:
            #     post_rot = post_rots[cam_i][:2, :2][None, ...]  # [1, 2, 2]
            #     post_tran = post_trans[cam_i][:2][None, ...]  # [1, 2]

            #     corners_2d_xy = lane_pts_2d[:, :, :2]
            #     corners_2d_z = lane_pts_2d[:, :, 2:3]
            #     corners_2d_xy = np.matmul(
            #         post_rot, np.transpose(corners_2d_xy, (0, 2, 1))
            #     ).transpose((0, 2, 1))
            #     corners_2d_xy = corners_2d_xy + post_tran[:, None, :]  # [N, 8, 2]
            #     lane_pts_2d = np.concatenate([corners_2d_xy, corners_2d_z], axis=-1)

            for lane_i in range(len(lane_pts_2d)):

                if lane_ignore[lane_i]:
                    continue

                pts_2d = lane_pts_2d[lane_i]  # [num_pts, 2]
                mask = (
                    (pts_2d[:, 0] > 0)
                    & (pts_2d[:, 0] < img_w)
                    & (pts_2d[:, 1] > 0)
                    & (pts_2d[:, 1] < img_h)
                    & (pts_2d[:, 2] > 0)
                )
                if mask.sum() == 0:
                    continue

                pts_2d = pts_2d[mask, :2]
                pts_2d = np.round(pts_2d).astype(np.int)

                if lane_grad_color[lane_i]:
                    colors = gradient_color(lane_colors[lane_i], _BLUE2, len(pts_2d))
                    for i in range(len(pts_2d) - 1):
                        color = colors[i % len(colors)]
                        cv2.line(img, pts_2d[i], pts_2d[i+1], color, 4*lane_sizes[lane_i])
                else:
                    cv2.polylines(
                        img, [pts_2d], False, lane_colors[lane_i], 3*lane_sizes[lane_i]
                    )
                    # pass

                if draw_point:
                    for point_i in range(len(pts_2d)):
                        pt = pts_2d[point_i]
                        cv2.circle(img, (int(pt[0]), int(pt[1])), 2, lane_colors[lane_i], lane_sizes[lane_i])


        draw_imgs.append(img)

    img_to_show = multicam_show_imgs(cams, draw_imgs)
    return img_to_show

def gradient_color(start_color, end_color, steps):
    return [tuple([(start_color[j] * (steps - i) + end_color[j] * i) // steps for j in range(3)])
            for i in range(steps)]

def gen_dx_bx_array(xbound, ybound, zbound, **kwargs):
    dx = np.array([row[2] for row in [xbound, ybound, zbound]])
    bx = np.array([row[0] for row in [xbound, ybound, zbound]])
    nx = np.array(
        [(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]], dtype=np.long
    )

    return dx, bx, nx

def draw_bev_ld(img_bev, map_res, draw_point=True):
    grid_config = map_res["grid_config"]
    lane_pts = map_res["lane_pts_3d"].copy()  # [lane_num, num_pts, 2]
    # lane_labels = map_res["lane_labels_3d"].copy()
    lane_colors = map_res["lane_colors"]
    lane_sizes = map_res["lane_sizes"]
    lane_grad_color = map_res["lane_grad_color"]
    lane_ignore = map_res["lane_ignore"]

    dx, bx, nx = gen_dx_bx_array(**grid_config)
    bev_painter_h, bev_painter_w = img_bev.shape[:2]

    ratio = bev_painter_h / int(nx[1])

    if len(lane_pts) == 0:
        return img_bev

    # lidar -> bev
    lane_pts[..., 0] = (lane_pts[..., 0] - bx[0]) / dx[0]
    lane_pts[..., 1] = (lane_pts[..., 1] - bx[1]) / dx[1]

    lane_pts = (lane_pts * ratio).astype(np.int)
    lane_pts[..., 1] = bev_painter_h - lane_pts[..., 1]

    for lane_i in range(len(lane_pts)):
        if lane_ignore[lane_i]:
            continue
        pts = lane_pts[lane_i]
        color = lane_colors[lane_i]
        thickness = lane_sizes[lane_i]
        # color = tuple(np.random.randint(0, 255, 3).tolist())
        if lane_grad_color[lane_i]:
            colors = gradient_color(lane_colors[lane_i], _BLUE2, len(pts))
            for i in range(len(pts) - 1):
                color = colors[i % len(colors)]
                cv2.line(img_bev, pts[i, :2], pts[i+1, :2], color, lane_sizes[lane_i])
        else:
            cv2.polylines(img_bev, [pts[..., :2]], False, color, thickness)

        if draw_point:
            for point_i in range(len(pts)):
                pt = pts[point_i]
                cv2.circle(img_bev, (int(pt[0]), int(pt[1])), 2, color, thickness)

    return img_bev

def draw_ld_vis(
    imgs,
    calib_infos,
    cams,
    img_bev,
    ratio,
    ld_infos=None,
    draw_aug_img=False,
    aug_imgs=None,
):

    lane_colors = []
    lane_sizes = []
    lane_grad_color = []
    lane_ignore = []
    lane_ignore_img = []
    for idx, label in enumerate(ld_infos["lane_labels_3d"]):
        cat = ld_infos["class_names"][label]
        color = ld_set_colors[cat]
        lane_colors.append(color)
        lane_sizes.append(2 if cat == 'waypoint' or cat == 'waypointGT' or cat == 'pw_waypoint' else 1)
        lane_grad_color.append(False if cat == 'waypoint' or cat == 'waypointGT' else False)
        # lane_ignore.append(True if cat == 'Center' else False)
        lane_ignore.append(False) # dummy
        lane_ignore_img.append(False if cat == 'waypoint' or cat == 'pw_waypoint' or cat == 'waypointInactive' or cat == 'pwwaypointInactive' else True)
        if (
            "lane_attributes" in ld_infos
            and "is_double_line" in ld_infos["lane_attributes"]
        ):
            if ld_infos["lane_attributes"]["is_double_line"][idx] == 1:
                lane_sizes[-1] *= 4
        if "lane_attributes" in ld_infos and "color" in ld_infos["lane_attributes"]:
            if ld_infos["lane_attributes"]["color"][idx] == 1:
                lane_colors[-1] = tuple([int(x * 0.5) for x in lane_colors[-1]])
    ld_infos["lane_colors"] = lane_colors
    ld_infos["lane_sizes"] = lane_sizes
    ld_infos["lane_grad_color"] = lane_grad_color
    ld_infos["lane_ignore"] = lane_ignore
    ld_infos["lane_ignore_img"] = lane_ignore_img

    img_bev = draw_bev_ld(img_bev, ld_infos, draw_point=False)
    img_to_show = draw_img_ld(imgs, calib_infos, cams, ld_infos, draw_point=False)

    if draw_aug_img:
        aug_img_to_show = draw_img_ld(
            aug_imgs, calib_infos, cams, ld_infos, is_aug=True
        )
        img_to_show = np.concatenate([aug_img_to_show, img_to_show], axis=1)

    return img_to_show, img_bev
