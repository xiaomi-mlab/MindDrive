import os
import os.path as osp
import av2.geometry.interpolate as interp_utils
import numpy as np
import copy
import cv2
import torch
import matplotlib.pyplot as plt
from PIL import Image
from shapely.geometry import LineString

def xywlr_to_vector(box):
    box = list(box.float().numpy())
    rect = ((box[0], box[1]), (box[2], box[3]), box[4] / np.pi * 180)
    box = cv2.boxPoints(rect)
    return box

def remove_nan_values(uv):
    is_u_valid = np.logical_not(np.isnan(uv[:, 0]))
    is_v_valid = np.logical_not(np.isnan(uv[:, 1]))
    is_uv_valid = np.logical_and(is_u_valid, is_v_valid)

    uv_valid = uv[is_uv_valid]
    return uv_valid

def points_ego2img(pts_ego, extrinsics, intrinsics, debug=False):
    pts_ego_4d = np.concatenate([pts_ego, np.ones([len(pts_ego), 1])], axis=-1)
    pts_cam_4d = extrinsics @ pts_ego_4d.T
    uv = (intrinsics @ pts_cam_4d[:3, :]).T

    uv = remove_nan_values(uv)
    depth = uv[:, 2]
    uv = uv[:, :2] / uv[:, 2].reshape(-1, 1)

    return uv, depth

def draw_polyline_ego_on_img(polyline_ego, img_bgr, extrinsics, intrinsics, color_bgr, thickness, debug=True):
    if polyline_ego.shape[1] == 2:
        zeros = np.zeros((polyline_ego.shape[0], 1))
        polyline_ego = np.concatenate([polyline_ego, zeros], axis=1)

    # if polyline_ego[0, 1] < polyline_ego[-1, 1]:
    #     polyline_ego = polyline_ego[::-1]

    # polyline_ego = interp_utils.interp_arc(t=500, points=polyline_ego)
    uv, depth = points_ego2img(polyline_ego, extrinsics, intrinsics, debug)
    h, w, c = img_bgr.shape


    is_valid_x = np.logical_and(0 <= uv[:, 0], uv[:, 0] < w - 1)
    is_valid_y = np.logical_and(0 <= uv[:, 1], uv[:, 1] < h - 1)
    is_valid_z = depth > 0
    is_valid_points = np.logical_and.reduce([is_valid_x, is_valid_y, is_valid_z])
    is_valid_points = np.logical_and(np.logical_and(is_valid_x, is_valid_y), is_valid_z)

    if is_valid_points.sum() == 0:
        return
    
    uv = np.round(uv[is_valid_points]).astype(np.int32)

    draw_visible_polyline_cv2(
        copy.deepcopy(uv),
        valid_pts_bool=np.ones((len(uv), 1), dtype=bool),
        image=img_bgr,
        color=color_bgr,
        thickness_px=thickness,
    )

def draw_visible_polyline_cv2(line, valid_pts_bool, image, color, thickness_px):
    """Draw a polyline onto an image using given line segments.

    Args:
        line: Array of shape (K, 2) representing the coordinates of line.
        valid_pts_bool: Array of shape (K,) representing which polyline coordinates are valid for rendering.
            For example, if the coordinate is occluded, a user might specify that it is invalid.
            Line segments touching an invalid vertex will not be rendered.
        image: Array of shape (H, W, 3), representing a 3-channel BGR image
        color: Tuple of shape (3,) with a BGR format color
        thickness_px: thickness (in pixels) to use when rendering the polyline.
    """
    line = np.round(line).astype(int)  # type: ignore
    for i in range(len(line) - 1):

        if (not valid_pts_bool[i]) or (not valid_pts_bool[i + 1]):
            continue

        x1 = line[i][0]
        y1 = line[i][1]
        x2 = line[i + 1][0]
        y2 = line[i + 1][1]

        # Use anti-aliasing (AA) for curves
        image = cv2.line(image, pt1=(x1, y1), pt2=(x2, y2), color=color, thickness=thickness_px, lineType=cv2.LINE_AA)

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

def draw_box_3d(image, corners, c=(0, 0, 255), direction_c=[0, 255, 0], extra_text=None, color=None):
    """
    :param image:  [256, 512, 3]
    :param corners: [8, 2], 2d corners
    :param c: colors
    :return:
    """

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
                         (int(corners[f[(j + 1) % 4], 0]), int(corners[f[(j + 1) % 4], 1])), c, 1, lineType=cv2.LINE_AA)
        if ind_f == 0:
            if valid[f[0]] and valid[f[2]]:
                cv2.line(image, (int(corners[f[0], 0]), int(corners[f[0], 1])),
                         (int(corners[f[2], 0]), int(corners[f[2], 1])), direction_c, 1, lineType=cv2.LINE_AA)
            if valid[f[1]] and valid[f[3]]:
                cv2.line(image, (int(corners[f[1], 0]), int(corners[f[1], 1])),
                         (int(corners[f[3], 0]), int(corners[f[3], 1])), direction_c, 1, lineType=cv2.LINE_AA)

    if color is not None:
        color = color
    else:
        color = (0, 0, 255)
    if extra_text:
        for i, txt in enumerate(extra_text.split('\n')):
            if txt:
                cv2.putText(image,txt,(int(min(corners[:,0])),int(min(corners[:,1])-10+20*i)),cv2.FONT_HERSHEY_SIMPLEX,0.75,color,2)
    return image


COLOR_MAPS_BGR = {
    # rbg colors
    'divider': (0, 0, 255),
    'boundary': (0, 255, 0),
    'ped_crossing': (255, 0, 0),
    'centerline': (51, 183, 255),
    'drivable_area': (171, 255, 255),
    'SolidLine': (255, 0, 0),
    'dotted_line': (0, 255, 0),
    'DottedLine': (0, 255, 0),
    'AlongRoadLine': (0, 0, 255),
    'ZebraCrossing': (51, 183, 255),
    'StopLine': (171, 255, 255),
    'OtherLine': (1, 1, 1),
    'CenterLine': (255, 102, 0),
    'ArrowLine': (255, 192, 203),
    'junction': (171, 255, 255),
}

## rgb
COLOR_MAPS_PLT = {
    'divider': 'r',
    'boundary': 'g',
    'ped_crossing': 'b',
    'centerline': 'orange',
    'drivable_area': 'y',
    'SolidLine': 'r',
    'dotted_line': 'g',
    'AlongRoadLine': 'b',
    'ZebraCrossing': 'orange',
    'StopLine': 'y',
    'OtherLine': 'white',
    'CenterLine': 'orange',
    'ArrowLine': 'pink',
    'junction': 'y'
}


CAM_NAMES_AV2 = ['ring_front_center', 'ring_front_right', 'ring_front_left',
    'ring_rear_right','ring_rear_left', 'ring_side_right', 'ring_side_left',
    ]
CAM_NAMES_NUSC = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',]

class Renderer(object):
    """Render map elements on image views.

    Args:
        cat2id (dict): category to class id
        roi_size (tuple): bev range
        dataset (str): 'av2' or 'nusc'
    """

    def __init__(self, 
                 roi_size=None, 
                 front_offset=0, 
                 grid_config=None, 
                 render_cfg={
                    'render_state' : False,
                    'render_expert' : True,
                 }):
        self.roi_size = roi_size
        self.front_offset = front_offset
        self.grid_config = grid_config
        self.render_cfg = render_cfg
        if self.grid_config is None:
            self.grid_config = {
                        "xbound": [-48, 48, 0.8],
                        "ybound": [-112, 144, 0.8],
                        "zbound": [-10.0, 10.0, 20.0],
                        "dbound": [1.0, 150.0, 1.0],
                    }

    def render_bev_from_vectors(self, vectors, road_flag, out_dir, idx, draw_scores=False, return_img=False):
        '''Render bev segmentation using vectorized map elements.
        
        Args:
            vectors (dict): dict of vectorized map elements.
            out_dir (str): output directory
        '''

        car_img = Image.open('.lidar_car.png')

        plt.figure(figsize=(self.roi_size[0], self.roi_size[1] + self.front_offset), dpi=10)
        plt.xlim(-self.roi_size[0] / 2, self.roi_size[0] / 2)
        plt.ylim(-self.roi_size[1] / 2, self.roi_size[1] / 2 + self.front_offset)
        plt.axis('off')
        plt.imshow(car_img, extent=[-2.5, 2.5, -2.0, 2.0])

        for label, vector_list in vectors.items():
            cat = self.id2cat[label]
            color = COLOR_MAPS_PLT[cat]
            for vector in vector_list:
                if draw_scores:
                    vector, score, prop = vector
                if isinstance(vector, list):
                    vector = np.array(vector)
                    vector = np.array(LineString(vector).simplify(0.2).coords)
                elif cat == 'StopLine' or LineString(vector).length < 6:
                    vector = np.array(LineString(vector).simplify(0.4).coords)

                pts = vector[:, :2]
                x = np.array([pt[0] for pt in pts])
                y = np.array([pt[1] for pt in pts])
                # plt.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1], angles='xy', color=color,
                #     scale_units='xy', scale=1)
                # for i in range(len(x)):
                plt.plot(x, y, 'o-', color=color, linewidth=20, markersize=50)
                if draw_scores:
                    if prop:
                        p = 'p'
                    else:
                        p = ''
                    score = round(score, 2)
                    mid_idx = len(x) // 2
                    plt.text(x[mid_idx], y[mid_idx], str(score)+p, fontsize=100, color=color)
        
        if road_flag is not None:
            for box in road_flag:
                vector = xywlr_to_vector(box)
                # vector = box
                pts = vector[:, :2]
                x = np.array([pt[0] for pt in pts])
                y = np.array([pt[1] for pt in pts])
                plt.plot(x, y, 'o-', color=color, linewidth=20, markersize=50)

        if not return_img:
            map_path = os.path.join(out_dir, f'map_{str(idx).replace("/", "-")}.jpg')
            plt.savefig(map_path, bbox_inches='tight', dpi=40)
            plt.close()
        else:
            from io import BytesIO
            buffer_ = BytesIO()
            plt.savefig(buffer_,format = 'png')
            buffer_.seek(0)
            dataPIL = Image.open(buffer_)
            ## rgb#
            data = np.asarray(dataPIL)[..., :3]
            buffer_.close()
            return data

    @staticmethod
    def gen_dx_bx(xbound, ybound, zbound, **kwargs):
        dx = np.array([row[2] for row in [xbound, ybound, zbound]])
        bx = np.array([row[0] for row in [xbound, ybound, zbound]])
        nx = np.array([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]], dtype=np.long)

        return dx, bx, nx

    def render_imgs_and_annos(self, imgs, bboxes, lanes, waypoints, extrinsics, intrinsics):
        if len(bboxes) > 0:
            bboxes = np.concatenate([box for box in bboxes.values()], axis=0)
        else:
            bboxes = np.empty((0, 7))
        surround_img = self.render_camera_views_from_vectors(lanes, waypoints, bboxes, imgs, 
                extrinsics, intrinsics)
        h = surround_img.shape[0]
        bev_img = self.render_bev_from_vectors_v2(lanes, bboxes, waypoints, bev_h=h)
        save_img = np.concatenate([surround_img, bev_img], axis=1)
        return save_img

    @staticmethod
    def gen_bev_painter(bev_painter_h, bev_painter_w, grid_config, ratio, background=255):
        img_bev = np.full((bev_painter_h, bev_painter_w, 3), background, dtype=np.uint8)

        x_min, x_max, x_interval = grid_config["xbound"]
        y_min, y_max, y_interval = grid_config["ybound"]

        for x in range(int(x_min) // 10 * 10, int(x_max) // 10 * 10 + 1, 10):
            canvas_bev_x = int((x - x_min) / x_interval * ratio)
            canvas_bev_y = int(bev_painter_h)
            cv2.line(img_bev, (canvas_bev_x, canvas_bev_y),
                    (canvas_bev_x, canvas_bev_y - 10), (255, 255, 255), 2)
            cv2.putText(img_bev, str(x) + "m", (canvas_bev_x - 10, canvas_bev_y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        for y in range(int(y_min) // 10 * 10, int(y_max) // 10 * 10 + 1, 10):
            canvas_bev_x = bev_painter_w
            canvas_bev_y = bev_painter_h - int((y - y_min) / y_interval * ratio)
            cv2.line(img_bev, (canvas_bev_x, canvas_bev_y),
                    (canvas_bev_x - 10, canvas_bev_y), (255, 255, 255), 2)
            cv2.putText(img_bev, str(y) + "m", (canvas_bev_x - 40, canvas_bev_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        return img_bev

    def render_bev_from_vectors_v2(self, vectors, bboxes=None, waypoints=None, bev_h=None, out_dir=None, idx=None, draw_scores=False, return_img=True):
        '''Render bev segmentation using vectorized map elements.
        
        Args:
            vectors (dict): dict of vectorized map elements.
            out_dir (str): output directory
        '''
        # car_img = np.array(Image.open('rl_projects/utils/lidar_car.png'))
        # car_img = cv2.resize(car_img, (9, 32))[:, :, :3]

        dx, bx, nx = self.gen_dx_bx(**self.grid_config)
        ratio = bev_h / int(nx[1])
        bev_painter_h = bev_h
        
        bev_painter_w = int(nx[0] * ratio)
        hdmap = self.gen_bev_painter(bev_painter_h, bev_painter_w, self.grid_config, ratio)

        # draw selfcar
        selfcar_position = np.array([[-1, 4], [1, -1]])
        selfcar_bev_x = ((selfcar_position[:, 0] - bx[0]) / dx[0] * ratio).astype(np.int)
        selfcar_bev_y = bev_painter_h - ((selfcar_position[:, 1] - bx[1]) / dx[1] * ratio).astype(np.int)
        cv2.rectangle(hdmap, (selfcar_bev_x[0], selfcar_bev_y[0]),
                    (selfcar_bev_x[1], selfcar_bev_y[1]), (0, 0, 0), -1)
        # hdmap[selfcar_bev_y[0]:selfcar_bev_y[1], selfcar_bev_x[0] : selfcar_bev_x[1], :] = car_img

        for label, vector_list in vectors.items():
            color = COLOR_MAPS_BGR[label]
            for line_ in vector_list:
                line = copy.deepcopy(line_)
                line[:, 0] = (line[:, 0] - bx[0]) / dx[0]
                line[:, 1] = (line[:, 1] - bx[1]) / dx[1]
                line = (line * ratio).astype(np.int)
                line[:, 1] = bev_painter_h - line[:, 1]
                cv2.polylines(hdmap, [line], False, color, 3)

        ################ draw boxes #################
        if bboxes is not None:
            corners_3d_lidar = gen_3d_object_corners_array(location=bboxes[:, :3],
                                                            dimension=bboxes[:, 3:6],
                                                            yaw=bboxes[:, 6])
            corners_3d_lidar = np.concatenate([corners_3d_lidar,
                                                np.ones((corners_3d_lidar.shape[0], 8, 1))], axis=2)
            corners_bev_lidar = corners_3d_lidar[:, :4, :2]
            for bbox_i in range(corners_bev_lidar.shape[0]):
                pts_bev = corners_bev_lidar[bbox_i, ...]
                pts_bev[:, 0] = (pts_bev[:, 0] - bx[0]) / dx[0]
                pts_bev[:, 1] = (pts_bev[:, 1] - bx[1]) / dx[1]

                pts_center = pts_bev.mean(0)
                # velo = bboxes[bbox_i, 7:9]
                # next_pts_center = pts_center + velo

                pts_bev = (pts_bev * ratio).astype(np.int)
                pts_bev[:, 1] = bev_painter_h - pts_bev[:, 1]

                # pts_center = (pts_center * ratio).astype(np.int)
                # pts_center[1] = bev_painter_h - pts_center[1]
                # next_pts_center = (next_pts_center * ratio).astype(np.int)
                # next_pts_center[1] = bev_painter_h - next_pts_center[1]
                
                color = (0, 255, 0)
                cv2.polylines(hdmap, [pts_bev], True, color, 2)

                # velo = bboxes[bbox_i, 7:9]
                # if velo.max() > 1000:
                #     cv2.polylines(hdmap, [pts_bev], True, _PURPLE, 2)

                # temp zoom velo vis
                # temp_velo_zoom_ratio = 2
                # draw_next_pts_center = (next_pts_center + pts_center) // temp_velo_zoom_ratio
                # if velo.max() > 1000:
                #     draw_next_pts_center = pts_center       # eval_online_show bbox
                # cv2.arrowedLine(img_bev, pts_center, draw_next_pts_center,
                #                 color=_RED, thickness=2)
        
        ## render waypoints
        if waypoints is not None:
            waypoints = waypoints[:12]
            waypoints = np.concatenate([np.zeros((1, 2)), waypoints], axis=0)
            wp_p = copy.deepcopy(waypoints)
            wp_p[:, 0] = (wp_p[:, 0] - bx[0]) / dx[0]
            wp_p[:, 1] = (wp_p[:, 1] - bx[1]) / dx[1]

            wp_p = (wp_p * ratio).astype(np.int)
            wp_p[:, 1] = bev_painter_h - wp_p[:, 1]
            cv2.polylines(hdmap, [wp_p], False, (0, 255, 0), 3)

            # line_left = copy.deepcopy(wp_p)
            # line_right = copy.deepcopy(wp_p)
            # line_right[:, 0] = line_right[:, 0] + 1.5 * ratio
            # line_left[:, 0] = line_left[:, 0] - 1.5 * ratio
            # line_right = line_right[::-1]
            # lines = np.concatenate([line_left, line_right], axis = 0)
            # mask = np.zeros(hdmap.shape, np.uint8)

            # # mask = cv2.polylines(mask, [lines], True, (0, 255, 0), 2)
            # # mask = cv2.fillPoly(mask, [lines], (255, 0, 0))  # 用于求 ROI
            # mask = cv2.fillPoly(mask, [lines], (0, 255, 0))
            # hdmap = cv2.addWeighted(hdmap, 0.5, mask, 0.5, 1)

        if not return_img:
            map_path = os.path.join(out_dir, f'map_{str(idx).replace("/", "-")}.jpg')
            cv2.imwrite(map_path, hdmap)
        else:
            return hdmap

    def render_camera_views_from_vectors(self, vectors, waypoints, gt_bboxes_3d, imgs, extrinsics, 
            intrinsics, thickness=2, out_dir=None, idx=None, return_img=True, extra_txt=None):
        if not isinstance(imgs[0], np.ndarray):
            imgs = [np.array(img) for img in imgs]
        h, w, _ = imgs[0].shape

        if gt_bboxes_3d is not None:
            if isinstance(gt_bboxes_3d, torch.Tensor):
                gt_bboxes_3d = gt_bboxes_3d.numpy()
            corners_3d_lidar = gen_3d_object_corners_array(location=gt_bboxes_3d[:, :3],
                                                        dimension=gt_bboxes_3d[:, 3:6],
                                                        yaw=gt_bboxes_3d[:, 6])
            corners_3d_lidar = np.concatenate([corners_3d_lidar,
                                            np.ones((corners_3d_lidar.shape[0], 8, 1))], axis=2)
        if len(imgs) == 7:
            all_img_show = np.ones(((h * 3, w * 3, 3))) * 255
        elif len(imgs) == 8:
            all_img_show = np.ones(((h * 4, w * 3, 3))) * 255
        else:
            all_img_show = np.ones(((h * 2, w * 3, 3))) * 255
        for i in range(len(imgs)):
            img = imgs[i]
            extrinsic = extrinsics[i]
            intrinsic = intrinsics[i]
            img_bgr = copy.deepcopy(img)
            
            for label, vector_list in vectors.items():
                debug = False
                if label == "DottedLine" and i == 6:
                    debug = True
                color = COLOR_MAPS_BGR[label]
                for vector in vector_list:
                    img_bgr = np.ascontiguousarray(img_bgr)
                    if isinstance(vector, list):
                        vector = np.array(vector).astype(float)
                    draw_polyline_ego_on_img(vector, img_bgr, extrinsic, intrinsic, 
                        color, thickness, debug=debug)

            # if road_flag is not None:
            #     for box in road_flag:
            #         vector = xywlr_to_vector(box)
            #         # vector = box
            #         draw_polyline_ego_on_img(vector, img_bgr, extrinsic, intrinsic, 
            #             (0, 255, 0), thickness)

            if gt_bboxes_3d is not None:
                ### draw box
                img_bgr = np.ascontiguousarray(img_bgr)
                img_h, img_w = img_bgr.shape[:2]

                corners_3d_cam = np.matmul(extrinsic[np.newaxis, ...],
                                        corners_3d_lidar.transpose((0, 2, 1)))
                corners_2d = np.matmul(intrinsic[np.newaxis, ...],
                                    corners_3d_cam[:, :3, :]).transpose((0, 2, 1))
                corners_2d[:, :, :2] = corners_2d[:, :, :2] / corners_2d[:, :, 2:3]

                inds = (corners_2d[:, :, 0] > 0) & (corners_2d[:, :, 0] < img_w) & \
                    (corners_2d[:, :, 1] > 0) & (corners_2d[:, :, 1] < img_h) & \
                    (corners_2d[:, :, 2] > 0)
                valid_mask = inds.sum(1) >= 4

                corners_2d = corners_2d[valid_mask, ...]
                corners_2d = corners_2d[:, :, :2]

                for bbox_i in range(corners_2d.shape[0]):
                    draw_box_3d(img_bgr, corners_2d[bbox_i], c=[0, 255, 0], extra_text=None, color=[0, 255, 0])

            if len(imgs) == 6:
                if i < 3:
                    all_img_show[:h, w * i : w * (i + 1), :] = img_bgr
                else:
                    all_img_show[h:, w * (i - 3) : w * (i - 2), :] = img_bgr
            elif len(imgs) == 7:
                if i < 3:
                    all_img_show[h:2*h, w * i : w * (i + 1), :] = img_bgr
                elif i >= 3 and i <= 5 :
                    all_img_show[2*h:, w * (i - 3) : w * (i - 2), :] = img_bgr
                else:
                    all_img_show[:h, w : 2*w, :] = img_bgr
            elif len(imgs) == 8:
                if i < 3:
                    all_img_show[h:2*h, w * i : w * (i + 1), :] = img_bgr
                elif i >= 3 and i <= 5 :
                    all_img_show[2*h:3*h, w * (i - 3) : w * (i - 2), :] = img_bgr
                elif i == 6:
                    all_img_show[:h, w : 2*w, :] = img_bgr
                else:
                    all_img_show[3*h : 4*h, w : 2*w, :] = img_bgr

            # out_path = osp.join(out_dir, self.cam_names[i]) + '.jpg'
            # cv2.imwrite(out_path, img_bgr)
        if extra_txt is not None:
            cv2.putText(all_img_show, extra_txt[0][1]['value'][:30], [20, 40], cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)
            cv2.putText(all_img_show, extra_txt[0][1]['value'][30:], [20, 80], cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)
            if len(extra_txt) > 1:
                cv2.putText(all_img_show, extra_txt[1][1]['value'], [20, 120], cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0), 2)
        if not return_img:
            out_path = osp.join(out_dir, f"surround_{str(idx).replace('/', '-')}" + '.jpg')
            cv2.imwrite(out_path, all_img_show)
        else:
            return all_img_show

    def render_bev_from_mask(self, semantic_mask, out_dir):
        '''Render bev segmentation from semantic_mask.
        
        Args:
            semantic_mask (array): semantic mask.
            out_dir (str): output directory
        '''

        c, h, w = semantic_mask.shape
        bev_img = np.ones((3, h, w), dtype=np.uint8) * 255
        if 'drivable_area' in self.cat2id:
            drivable_area_mask = semantic_mask[self.cat2id['drivable_area']]
            bev_img[:, drivable_area_mask == 1] = \
                    np.array(COLOR_MAPS_BGR['drivable_area']).reshape(3, 1)

        for label in range(c):
            cat = self.id2cat[label]
            if cat == 'drivable_area':
                continue
            mask = semantic_mask[label]
            valid = mask == 1
            bev_img[:, valid] = np.array(COLOR_MAPS_BGR[cat]).reshape(3, 1)
        
        bev_img_flipud = np.array([np.flipud(i) for i in bev_img], dtype=np.uint8)
        out_path = osp.join(out_dir, 'semantic_map.jpg')
        cv2.imwrite(out_path, bev_img_flipud.transpose((1, 2, 0)))
        
