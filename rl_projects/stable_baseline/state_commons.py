import torch
from torchvision import transforms

import numpy as np
import gymnasium as gym

from rl_projects.envs.carla_env.carla_wrappers import vector, get_displacement_vector

torch.cuda.empty_cache()

def preprocess_frame(frame):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
    ])
    frame = preprocess(frame).unsqueeze(0)
    return frame


def create_encode_state_fn_gs(CONFIG):
    observation_space = {}
    observation_space['render_camera'] = gym.spaces.Box(low=0, high=255, shape=(8, 540, 960, 3), dtype=np.uint8)
    observation_space['render_points'] = gym.spaces.Box(low=0, high=255, shape=(10000, 4), dtype=np.float32)
    observation_space['ego_pose'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4, 4), dtype=np.float32)
    observation_space['extrinsics'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(8, 4, 4), dtype=np.float32)
    observation_space['intrinsics'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(8, 3, 3), dtype=np.float32)
    observation_space['timestamp'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, ), dtype=np.float64)
    observation_space['clip_first_flag'] = gym.spaces.Box(low=0, high=1, shape=(1, ), dtype=np.int)
    observation_space['ego_fut_cmd'] = gym.spaces.Box(low=0, high=1, shape=(3, ), dtype=np.int)
    observation_space['data_idx'] = gym.spaces.Box(low=0, high=np.inf, shape=(1, ), dtype=np.int)
    return gym.spaces.Dict(observation_space), None

def create_encode_state_fn(measurements_to_include, CONFIG, vae=None):
    """
        Returns a function that encodes the current state of
        the environment into some feature vector.
    """

    measure_flags = ["steer" in measurements_to_include, # True
                     "throttle" in measurements_to_include, # True
                     "speed" in measurements_to_include, # True
                     "angle_next_waypoint" in measurements_to_include,
                     "maneuver" in measurements_to_include,
                     "waypoints" in measurements_to_include,# True
                     "rgb_camera" in measurements_to_include,
                     "seg_camera" in measurements_to_include, # True
                     "end_wp_vector" in measurements_to_include,
                     "end_wp_fixed" in measurements_to_include,
                     "distance_goal" in measurements_to_include]

    def create_observation_space():
        observation_space = {}
        low, high = [], []
        if measure_flags[0]: low.append(-1), high.append(1)
        if measure_flags[1]: low.append(0), high.append(1)
        if measure_flags[2]: low.append(0), high.append(120)
        if measure_flags[3]: low.append(-3.14), high.append(3.14)
        observation_space['vehicle_measures'] = gym.spaces.Box(low=np.array(low), high=np.array(high), dtype=np.float32)

        if measure_flags[4]: observation_space['maneuver'] = gym.spaces.Discrete(4)

        if measure_flags[5]: observation_space['waypoints'] = gym.spaces.Box(low=-50, high=50, shape=(15, 2),
                                                                             dtype=np.float32)

        if measure_flags[6]: observation_space['rgb_camera'] = gym.spaces.Box(low=0, high=255, shape=(CONFIG.obs_res[1], CONFIG.obs_res[0], 3), dtype=np.uint8)
        if measure_flags[7]: observation_space['seg_camera'] = gym.spaces.Box(low=0, high=255, shape=(CONFIG.obs_res[1], CONFIG.obs_res[0], 3), dtype=np.uint8)
        if measure_flags[8]: observation_space['end_wp_vector'] = gym.spaces.Box(low=-50, high=50, shape=(1, 2), dtype=np.float32)
        if measure_flags[9]: observation_space['end_wp_fixed'] = gym.spaces.Box(low=-50, high=50, shape=(1, 2), dtype=np.float32)
        if measure_flags[10]: observation_space['distance_goal'] = gym.spaces.Box(low=0, high=1500, shape=(1, 1), dtype=np.float32)

        if CONFIG.use_seg_bev: observation_space['seg_camera'] = gym.spaces.Box(low=0, high=255, shape=(192, 192, 6), dtype=np.uint8)

        return gym.spaces.Dict(observation_space)

    def encode_state(env):
        encoded_state = {}
        vehicle_measures = []
        if measure_flags[0]: vehicle_measures.append(env.vehicle.control.steer)
        if measure_flags[1]: vehicle_measures.append(env.vehicle.control.throttle)
        if measure_flags[2]: vehicle_measures.append(env.vehicle.get_speed())
        if measure_flags[3]: vehicle_measures.append(env.vehicle.get_angle(env.current_waypoint))
        encoded_state['vehicle_measures'] = vehicle_measures
        if measure_flags[4]: encoded_state['maneuver'] = env.current_road_maneuver.value

        if measure_flags[5]:
            next_waypoints_state = env.route_waypoints[env.current_waypoint_index: env.current_waypoint_index + 15]
            waypoints = [vector(way[0].transform.location) for way in next_waypoints_state]

            vehicle_location = vector(env.vehicle.get_location())
            theta = np.deg2rad(env.vehicle.get_transform().rotation.yaw)

            relative_waypoints = np.zeros((15, 2))
            for i, w_location in enumerate(waypoints):
                relative_waypoints[i] = get_displacement_vector(vehicle_location, w_location, theta)[:2]
            if len(waypoints) < 15:
                start_index = len(waypoints)
                reference_vector = relative_waypoints[start_index-1] - relative_waypoints[start_index-2]
                for i in range(start_index, 15):
                    relative_waypoints[i] = relative_waypoints[i-1] + reference_vector

            encoded_state['waypoints'] = relative_waypoints

        if measure_flags[6]: encoded_state['rgb_camera'] = env.observation
        if measure_flags[7] : encoded_state['seg_camera'] = env.observation
        if measure_flags[8]:
            vehicle_location = vector(env.vehicle.get_location())
            theta = np.deg2rad(env.vehicle.get_transform().rotation.yaw)
            end_wp_location = vector(env.end_wp.transform.location)
            encoded_state['end_wp_vector'] = get_displacement_vector(vehicle_location, end_wp_location, theta)[:2]
        if measure_flags[9]:
            vehicle_location = vector(env.start_wp.transform.location)
            theta = np.deg2rad(env.start_wp.transform.rotation.yaw)
            end_wp_location = vector(env.end_wp.transform.location)
            encoded_state['end_wp_fixed'] = get_displacement_vector(vehicle_location, end_wp_location, theta)[:2]
        if measure_flags[10]:
            encoded_state['distance_goal'] = [[len(env.route_waypoints) - env.current_waypoint_index]]

        return encoded_state

    return create_observation_space(), encode_state
