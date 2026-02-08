import torch as th
import numpy as np
import torch.nn as nn
import gymnasium as gym
import torch
from ..utils.utils import lr_schedule

from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import get_flattened_obs_dim

class AttrDict:
    def __init__(self, data):
        self.__dict__.update(data)

algorithm_params = {
    ## for carla env
    "PPO": dict(
        device="cuda:0",
        learning_rate=lr_schedule(1e-4, 1e-6, 2),
        gamma=0.98,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,
        n_epochs=10,
        n_steps=1024, ## NOTE
    ),
    "SAC": dict(
        device="cuda:0",
        learning_rate=lr_schedule(5e-4, 1e-6, 2),
        buffer_size=100000,
        batch_size=256,
        ent_coef='auto',
        gamma=0.98,
        tau=0.02,
        train_freq=64,
        gradient_steps=64,
        learning_starts=10000,
        use_sde=True,
        policy_kwargs=dict(log_std_init=-3, net_arch=[400, 300]),
    ),
    ### for 3DGS env
    "PPO_GS": dict(
        device="cuda:0",
        learning_rate=lr_schedule(1e-4, 1e-6, 2),
        gamma=0.98,
        batch_size=4,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,
        n_epochs=10,
        n_steps=16, ## NOTE
    ),
}

states = {
    "1": ["steer", "throttle", "speed", "angle_next_waypoint", "maneuver"],
    "2": ["steer", "throttle", "speed", "maneuver"],
    "3": ["steer", "throttle", "speed", "waypoints"],
    "4": ["steer", "throttle", "speed", "angle_next_waypoint", "maneuver", "distance_goal"],
    "5": ["steer", "throttle", "speed", "waypoints", "seg_camera"],
}

reward_params = {
    "reward_rule": dict(
        early_stop=True,
        min_speed=3.6 / 3.6,  # km/h
        max_speed=72.0 / 3.6,  # km/h
        target_speed=36.0 / 3.6,  # kmh
        max_distance=2.0,  # Max distance from center before terminating
        max_std_center_lane=0.4,
        max_angle_center_lane=40,
        penalty_reward=-10,
    )
}

_CONFIG_rule_ppo = {
    "algorithm": "RULE-PPO",
    "algorithm_params": AttrDict(algorithm_params["PPO"]),
    "state": states["5"],
    "action_smoothing": 0.75,
    "reward_fn": "reward_rule",
    "reward_params": AttrDict(reward_params["reward_rule"]),
    "obs_res": (80, 120),
    "seed": 100,
    "wrappers": [],
    "action_noise": {},
    "action_space_type": "discrete",
    "action_type" : "plan",
    "use_seg_bev": True,
    "use_rgb_bev": True,
}

_CONFIG_gs_ppo = {
    "algorithm": "GS-PPO",
    "algorithm_params": algorithm_params["PPO_GS"],
    "state": states["5"],
    "action_smoothing": 0.75,
    "reward_fn": "reward_rule",
    "reward_params": reward_params["reward_rule"],
    "obs_res": (80, 120),
    "seed": 100,
    "wrappers": [],
    "action_noise": {},
    "action_space_type": "discrete",
    "action_type" : "plan",
    "use_seg_bev": True,
    "use_rgb_bev": True,
}


CONFIGS = {
    "rule_ppo": _CONFIG_rule_ppo,
    "gs_ppo": _CONFIG_gs_ppo,
}

CONFIG = AttrDict(_CONFIG_rule_ppo)

# def set_config(config_name):
#     global CONFIG
#     CONFIG = CONFIGS[config_name]
#     return CONFIG
