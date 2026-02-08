import mmcv
import os
import cv2
import pickle
import numpy as np
from scipy.spatial.transform import Rotation as R

def project_pinhole_distort(points, ck, cdist):
    fx, fy, cx, cy = ck
    x = points[0]
    y = points[1]
    z = points[2]
    if len(cdist) == 4:
        cdist.append(0)

    k1, k2, p1, p2, k3 = cdist
    xu =  x / z
    yu = y / z
    r2 = xu * xu + yu * yu
    r4 = r2 * r2
    r6 = r2 * r2 * r2
    xd = (1 + k1 * r2 + k2 * r4 + k3 * r6) * xu + 2 * p1 * xu * yu + p2 * (r2 + 2 * xu * xu)
    yd = (1 + k1 * r2 + k2 * r4 + k3 * r6) * yu + 2 * p2 * xu * yu + p1 * (r2 + 2 * yu * yu)
    u = fx * xd + cx
    v = fy * yd + cy
    return u, v

def project_pinhole(points, ck):
    fx, fy, cx, cy = ck
    x = points[0]
    y = points[1]
    z = points[2]
    xu = x / z
    yu = y / z
    u = fx * xu + cx
    v = fy * yu + cy
    return u, v


def project_fisheye_omni(points, alpha, ck, cdist):
    x, y, z = points[0], points[1], points[2]
    d = np.sqrt(x * x + y * y + z * z)
    rz = z + alpha * d
    x = x / rz
    y = y / rz
    xx = x * x
    yy = y * y
    xy = x * y
    theta2 = xx + yy
    distortion = cdist[0] * theta2 + cdist[1] * theta2 * theta2
    x = x + x * distortion + 2.0 * cdist[2] * xy + cdist[3] * (theta2 + 2.0 * xx)
    y = y + y * distortion + 2.0 * cdist[3] * xy + cdist[2] * (theta2 + 2.0 * yy)
    x = ck[0] * x + ck[2]
    y = ck[1] * y + ck[3]
    return x, y


def project_kannala_brandt(points, ck, cdist):
    x, y, z = points[0], points[1], points[2]
    x = x / z
    y = y / z

    r = np.sqrt(x**2 + y**2)
    theta = np.arctan(r)
    theta_2 = theta * theta
    theta_4 = theta_2 * theta_2
    theta_6 = theta_4 * theta_2
    theta_8 = theta_4 * theta_4
    theta_d = theta * (1 + cdist[0] * theta_2 + cdist[1] * theta_4 + cdist[2] * theta_6 +
                       cdist[3] * theta_8)

    x = (theta_d / r) * x
    y = (theta_d / r) * y
    x = ck[0] * x + ck[2]
    y = ck[1] * y + ck[3]

    return x, y

def warp_pinhole(w,
                 h,
                 intrinsic,
                 distortions,
                 sensor_to_vehicle,
                 v_sensor_to_vehicle_,
                 intrinsic_v=[0, 0, 0, 0],
                 distortions_v=[0, 0, 0, 0],
                 type='omni'):
    fx_v, fy_v, cx_v, cy_v = intrinsic_v
    us, vs = np.meshgrid(np.arange(int(w)), np.arange(int(h)))
    x_ = (us - cx_v) / fx_v
    y_ = (vs - cy_v) / fy_v
    z_ = np.ones_like(x_)

    rays = np.hstack((x_.reshape((-1, 1)), y_.reshape((-1, 1)), z_.reshape((-1, 1))))
    rays = np.matmul(v_sensor_to_vehicle_[:3, :3], rays.transpose())
    rays = np.matmul(np.linalg.inv(sensor_to_vehicle[:3, :3]), rays)
    x_ = rays[0, :]
    y_ = rays[1, :]
    z_ = rays[2, :]
    if type == 'omni':
        u, v = project_fisheye_omni([x_, y_, z_], intrinsic[0], intrinsic[1:], distortions)
    elif type == 'pinhole_distort':
        if(len(distortions) == 4):
            distortions.append(0)
        u, v = project_pinhole_distort([x_, y_, z_], intrinsic, distortions)
    elif type == 'kb':
        u, v = project_kannala_brandt([x_, y_, z_], intrinsic, distortions)
    else:
        u, v = project_pinhole([x_, y_, z_], intrinsic)

    map_x = u.reshape(int(h), int(w)).astype(np.float32)
    map_y = v.reshape(int(h), int(w)).astype(np.float32)
    return map_x, map_y

def gen_bhd_remap_calib_infos(cams, calib_infos, remap_shape, is_record=False):
    new_calib_infos = {}
    remap_W, remap_H = remap_shape
    for cam in cams:
        new_calib_info = {}

        if cam == "crop_f120":
            K = np.array(calib_infos["mid_center_top_wide"]["intrinsics"].copy())
            D = calib_infos["mid_center_top_wide"]["distortions"].copy()
            m = np.array(calib_infos["mid_center_top_wide"]['cam2lidar'].copy()).reshape((4, 4))
            ori_shape = calib_infos["mid_center_top_wide"]["ori_shape"]
            model = calib_infos["mid_center_top_wide"]["model"]
        elif cam == "crop_r60":
            K = np.array(calib_infos["rear_center_top_norm"]["intrinsics"].copy())
            D = calib_infos["rear_center_top_norm"]["distortions"].copy()
            m = np.array(calib_infos["rear_center_top_norm"]['cam2lidar'].copy()).reshape((4, 4))
            ori_shape = calib_infos["rear_center_top_norm"]["ori_shape"]
            model = calib_infos["rear_center_top_norm"]["model"]
        elif cam == "crop_f30" or cam == "crop_f30_2" or cam == "crop_f30_3":
            K = np.array(calib_infos["mid_center_top_tele"]["intrinsics"].copy())
            D = calib_infos["mid_center_top_tele"]["distortions"].copy()
            m = np.array(calib_infos["mid_center_top_tele"]['cam2lidar'].copy()).reshape((4, 4))
            ori_shape = calib_infos["mid_center_top_tele"]["ori_shape"]
            model = calib_infos["mid_center_top_tele"]["model"]
        else:
            K = np.array(calib_infos[cam]["intrinsics"])
            D = calib_infos[cam]["distortions"]
            m = np.array(calib_infos[cam]['cam2lidar']).reshape((4, 4))
            ori_shape = calib_infos[cam]["ori_shape"]
            model = calib_infos[cam]["model"]

        if is_record:
            K = K / 1.5
            ori_shape = (int(ori_shape[0] / 1.5), int(ori_shape[1] / 1.5))

        if cam in ["mid_center_top_wide", "front_left_bottom_wide", "front_right_bottom_wide",
                   "rear_left_bottom_wide", "rear_right_bottom_wide"]:
            assert model == "kannala-brandt" or model == "KannalaBrandtModel"
            remap_k = [0.45 * remap_W, 0.95 * remap_H, 0.5 * remap_W, 0.5 * remap_H]
            map_x, map_y = warp_pinhole(remap_W, remap_H, K, D, m, m, remap_k, type='kb')
        elif cam in ["crop_f120"]:
            remap_k = [1850, 1850, 486, 275]
            map_x, map_y = warp_pinhole(remap_W, remap_H, K, D, m, m, remap_k, type='pinhole_distort')
        elif cam in ['rear_center_top_norm']:  # rear_60
            remap_k = [0.8 * remap_W, 1.35 * remap_H, 0.5 * remap_W, 0.5 * remap_H]
            map_x, map_y = warp_pinhole(remap_W, remap_H, K, D, m, m, remap_k, type='pinhole_distort')
        elif cam in ["crop_r60"]:       # crop_rear_60
            remap_k = [2400, 2400, 480.0, 270.0]
            map_x, map_y = warp_pinhole(remap_W, remap_H, K, D, m, m, remap_k, type='pinhole_distort')
        elif cam in ["mid_center_top_tele"]:
            remap_k = [1850, 1850, 486, 275]
            map_x, map_y = warp_pinhole(remap_W, remap_H, K, D, m, m, remap_k, type='pinhole_distort')
            # ori_k = np.array([[K[0], 0, K[2]], [0, K[1], K[3]], [0, 0, 1]])
            # remap_k = cv2.getOptimalNewCameraMatrix(ori_k.copy(),
            #                                         distCoeffs=np.array(D),
            #                                         imageSize=ori_shape,
            #                                         alpha=0,
            #                                         newImgSize=remap_shape)[0]
            # map_x, map_y = cv2.initUndistortRectifyMap(ori_k,
            #                                            np.array(D),
            #                                            None,
            #                                            newCameraMatrix=remap_k,
            #                                            size=remap_shape,
            #                                            m1type=5)
            # remap_k = [remap_k[0][0], remap_k[1][1], remap_k[0][2], remap_k[1][2]]
        elif cam in ["crop_f30"]:
            remap_k = [1850, 1850, 486, 275]
            sensor_to_vehicle = m
            v_sensor_to_vehicle = np.array(calib_infos["mid_center_top_wide"]['cam2lidar'].copy()).reshape((4, 4))

            m = v_sensor_to_vehicle
            m[:3, 3] = sensor_to_vehicle[:3, 3]
            map_x, map_y = warp_pinhole(remap_W, remap_H, K, D, sensor_to_vehicle, v_sensor_to_vehicle, remap_k,
                                        type='pinhole_distort')
        elif cam in ["crop_f30_2"]:
            remap_k = [1850, 1850, 486, 275]
            map_x, map_y = warp_pinhole(remap_W, remap_H, K, D, m, m, remap_k, type='pinhole_distort')
        elif cam in ["crop_f30_3"]:
            remap_k = [1850, 1850, 486, 275]
            sensor_to_vehicle = m.copy()
            euler_angle = rotationMatrixToEulerAngles(sensor_to_vehicle[:3, :3])
            new_euler_angle = euler_angle.copy()
            new_euler_angle[0] = -np.pi / 2

            # print("euler_angle: {}, new_euler_angle: {}".format(
            #     euler_angle * 180 / np.pi, new_euler_angle * 180 / np.pi))
            roll, pitch, yaw = new_euler_angle
            new_rot_matrix = euler_to_rotMat(yaw, pitch, roll)

            v_sensor_to_vehicle = sensor_to_vehicle.copy()
            v_sensor_to_vehicle[:3, :3] = new_rot_matrix

            # print("sensor_to_vehicle: ", sensor_to_vehicle)
            # print("v_sensor_to_vehicle: ", v_sensor_to_vehicle)
            m = v_sensor_to_vehicle
            map_x, map_y = warp_pinhole(remap_W, remap_H, K, D, sensor_to_vehicle, v_sensor_to_vehicle, remap_k,
                                        type='pinhole_distort')

        new_calib_info["intrinsics"] = \
            np.array([remap_k[0], 0, remap_k[2], 0,
                      0, remap_k[1], remap_k[3], 0,
                      0, 0, 1, 0,
                      0, 0, 0, 1]).reshape(4, 4)
        new_calib_info["cam2lidar"] = m
        new_calib_info["lidar2cam"] = np.linalg.inv(m)
        new_calib_info["undistort_map_coors"] = [map_x, map_y]

        new_calib_infos[cam] = new_calib_info
    return new_calib_infos

def parse_bhd_ori_calib(sensor_infos):
    cam_model_dict = {
        "KannalaBrandtModel": "kb_model",
        "PinHoleModel": "pinhole_model",
    }
    cam_model_dict2 = {
        "KannalaBrandtModel": "kbModel",
        "PinHoleModel": "pinholeModel",
    }
    cam_infos = {}
    for cam_info in sensor_infos["camera"]:
        new_cam_info = {}
        width = cam_info["width"]
        height = cam_info["height"]
        fov = cam_info["fov"]
        cam_name = cam_info["name"]["label"].lower()
        if cam_name not in ["mid_center_top_tele",  # front 30
                             # "mid_center_top_norm",     # front 60
                             "rear_center_top_norm",  # rear 60
                             "mid_center_top_wide",  # front 120
                             "front_left_bottom_wide",  # front left 100
                             "rear_left_bottom_wide",  # rear left 100
                             "front_right_bottom_wide",  # front right 100
                             "rear_right_bottom_wide"]:  # rear right 100
            continue

        if "specific_model" in cam_info:
            cam_model_name = cam_info["specific_model"]['model']
            cam_params = cam_info['specific_model'][
                cam_model_dict[cam_model_name]]
        elif 'specificModel' in cam_info:
            cam_model_name = cam_info["specificModel"]['model']
            cam_params = cam_info['specificModel'][
                cam_model_dict2[cam_model_name]]
        else:
            raise Exception('no specific model')

        cam_K = []
        distortion = []
        cam_K = [
            cam_params["fu"], cam_params["fv"], cam_params["pu"],
            cam_params["pv"]
        ]
        # print("cam_name: {}, cam_model: {}".format(cam_name,
        #                                            cam_model_name))
        if cam_model_name == "PinHoleModel":
            distortion = [
                cam_params["distortionsK1"], cam_params["distortionsK2"],
                cam_params["distortionsP1"], cam_params["distortionsP2"],
                cam_params["distortionsK3"]
            ]
        elif cam_model_name == "KannalaBrandtModel":
            distortion = [
                cam_params["distortionsK1"], cam_params["distortionsK2"],
                cam_params["distortionsK3"], cam_params["distortionsK4"]
            ]
        else:
            raise Exception("unknown cam model")
        cam2vehicle_rq = [
            cam_info["sensorToVehicle"]["rotation"]["x"],
            cam_info["sensorToVehicle"]["rotation"]["y"],
            cam_info["sensorToVehicle"]["rotation"]["z"],
            cam_info["sensorToVehicle"]["rotation"]["w"]
        ]  # w, x, y, z
        cam2vehicle_tran = [
            cam_info["sensorToVehicle"]["translation"]["x"],
            cam_info["sensorToVehicle"]["translation"]["y"],
            cam_info["sensorToVehicle"]["translation"]["z"]
        ]  # x, y, z
        cam2vehicle_rm = R.from_quat(cam2vehicle_rq).as_matrix()
        cam2vehicle = np.eye(4)
        cam2vehicle[:3, :3] = cam2vehicle_rm
        cam2vehicle[:3, 3] = np.array(cam2vehicle_tran)

        new_cam_info["distortions"] = distortion
        new_cam_info["intrinsics"] = cam_K
        new_cam_info["cam2lidar"] = cam2vehicle.reshape(-1).tolist()
        new_cam_info["ori_shape"] = (width, height)
        new_cam_info['model'] = cam_model_name
        cam_infos[cam_name] = new_cam_info

    return cam_infos
