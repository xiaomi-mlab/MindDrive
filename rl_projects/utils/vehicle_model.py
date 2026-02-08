import copy
import numpy as np
from scipy.spatial.transform import Rotation
from ..envs.utils import rotation_matrix_to_euler

class VehicleModel(object):
    def __init__(self, dt=0.5, length=3):
        self.length = length
        self.dt = 0.5

    def update_vehicle_state(self, action, ego_pose, cur_canbus):
        cur_pos = cur_canbus[:3]
        cur_rot = cur_canbus[3:7]
        cur_acc = cur_canbus[7:9]
        cur_angule = cur_canbus[10:12]
        cur_speed = cur_canbus[13:15] # vx vy
        cur_yaw = cur_canbus[16]

        ### use pid????

        v = np.linalg.norm(np.array(action))
        x, y = ego_pose[0, 3], ego_pose[1, 3]
        _, _, theta = rotation_matrix_to_euler(ego_pose[:3, :3])

        dx = v * np.cos(theta) * self.dt
        dy = v * np.sin(theta) * self.dt
        dtheta = (v / self.length) * np.tan(cur_yaw) * self.dt
        
        # 更新车辆的位置和航向角
        x_new = x + dx
        y_new = y + dy
        theta_new = theta + dtheta

        next_pose = np.eye(4)
        rot = np.array([np.cos(theta_new), -np.sin(theta_new), 0,
                        np.sin(theta_new), np.cos(theta_new), 0,
                        0, 0, 1]).reshape(3, 3)
        next_pose[:3, :3] = rot
        next_pose[0, 3] = x_new
        next_pose[1, 3] = y_new

        ax, ay = 0, 0

        next_canbus = copy.deepcopy(cur_canbus)
        next_canbus[:2] = [x_new, y_new]
        next_canbus[3:7] = Rotation.from_matrix(rot).as_quat(scalar_first=True)
        next_canbus[7:9] = [ax, ay]
        next_canbus[10:12] = [0, 0]
        next_canbus[13:15] = [0, v]
        next_canbus[16] = -theta
        next_canbus[17] = -theta / np.pi * 180

        return next_pose, next_canbus
