import carla
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from rl_projects.scenario_runner.srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from rl_projects.leaderboard.scenarios.scenario_manager import ScenarioManager
from rl_projects.leaderboard.scenarios.route_scenario import RouteScenario
from rl_projects.leaderboard.utils.statistics_manager import StatisticsManager, FAILURE_MESSAGES
from rl_projects.leaderboard.utils.route_indexer import RouteIndexer 
from rl_projects.utils.pid_controller_decouple import PIDController
from rl_projects.scenario_runner.srunner.scenariomanager.timer import GameTime
from datetime import datetime
import traceback
from rl_projects.scenario_runner.srunner.scenariomanager.watchdog import Watchdog
from rl_projects.envs.carla_env.carla_wrappers import *
from loguru import logger
import threading
import time
import py_trees
from rl_projects.leaderboard.envs.sensor_interface import CallBack, SensorInterface, SpeedometerReader
from PIL import Image
import numpy as np
import os
import cv2
from pyquaternion import Quaternion
from team_code.planner import RoutePlanner
from scipy.optimize import fsolve
from rl_projects.leaderboard.utils.result_writer import ResultOutputProvider
from rl_projects.scenario_runner.srunner.scenariomanager.traffic_events import TrafficEventType
import random
import json
import pathlib
from rl_projects.leaderboard.utils.route_manipulation import downsample_route
from rl_projects.utils.carla_utils import kill_existing_carla, launch_carla
import socket

SAVE_PATH = os.environ.get('SAVE_PATH', None)

def find_free_port(starting_port):
    port = starting_port
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("localhost", port))
                return port
        except OSError:
            port += 1

class TickRuntimeError(Exception):
    pass

class CarlaScenarioEnv(gym.Env):

    def __init__(self, 
                 routes,
                 repetitions = 1,
                 host="127.0.0.1",
                 port=40000,
                 reward_fn=None,
                 action_smoothing=0.0,
                 action_space_type="continuous",
                 action_type="plan",
                 traffic_manager_port=48000,
                 traffic_manager_seed=0,
                 checkpoint=None,
                 resume = None,
                 max_tick_count=3000,
                 debug=False):
        
        self.routes = routes
        self.repetitions =  repetitions
        self.resume = resume
        self.routes_subset = ''
        self.host = host
        self.port = port
        self.traffic_manager_port = find_free_port(traffic_manager_port)
        self.save_path = None
        self.checkpoint = checkpoint
        self.debug = debug
        self._timeout = 300
        self._debug_mode = 1
        self.tick_count = 0
        self._sensors_list=[]
        self._timestamp_last_run = 0.0
        self.wallclock_t0=None
        self.max_tick_count=max_tick_count

        self.lidar2img = {
        'CAM_FRONT':np.array([[ 1.14251841e+03,  8.00000000e+02,  0.00000000e+00, -9.52000000e+02],
                                  [ 0.00000000e+00,  4.50000000e+02, -1.14251841e+03, -8.09704417e+02],
                                  [ 0.00000000e+00,  1.00000000e+00,  0.00000000e+00, -1.19000000e+00],
                                 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
        'CAM_FRONT_LEFT':np.array([[ 6.03961325e-14,  1.39475744e+03,  0.00000000e+00, -9.20539908e+02],
                                   [-3.68618420e+02,  2.58109396e+02, -1.14251841e+03, -6.47296750e+02],
                                   [-8.19152044e-01,  5.73576436e-01,  0.00000000e+00, -8.29094072e-01],
                                   [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
        'CAM_FRONT_RIGHT':np.array([[ 1.31064327e+03, -4.77035138e+02,  0.00000000e+00,-4.06010608e+02],
                                       [ 3.68618420e+02,  2.58109396e+02, -1.14251841e+03,-6.47296750e+02],
                                    [ 8.19152044e-01,  5.73576436e-01,  0.00000000e+00,-8.29094072e-01],
                                    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]]),
        'CAM_BACK':np.array([[-5.60166031e+02, -8.00000000e+02,  0.00000000e+00, -1.28800000e+03],
                     [ 5.51091060e-14, -4.50000000e+02, -5.60166031e+02, -8.58939847e+02],
                     [ 1.22464680e-16, -1.00000000e+00,  0.00000000e+00, -1.61000000e+00],
                     [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
        'CAM_BACK_LEFT':np.array([[-1.14251841e+03,  8.00000000e+02,  0.00000000e+00, -6.84385123e+02],
                                  [-4.22861679e+02, -1.53909064e+02, -1.14251841e+03, -4.96004706e+02],
                                  [-9.39692621e-01, -3.42020143e-01,  0.00000000e+00, -4.92889531e-01],
                                  [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
  
        'CAM_BACK_RIGHT': np.array([[ 3.60989788e+02, -1.34723223e+03,  0.00000000e+00, -1.04238127e+02],
                                    [ 4.22861679e+02, -1.53909064e+02, -1.14251841e+03, -4.96004706e+02],
                                    [ 9.39692621e-01, -3.42020143e-01,  0.00000000e+00, -4.92889531e-01],
                                    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
        }
        self.lidar2cam = {
        'CAM_FRONT':np.array([[ 1.  ,  0.  ,  0.  ,  0.  ],
                                 [ 0.  ,  0.  , -1.  , -0.24],
                                 [ 0.  ,  1.  ,  0.  , -1.19],
                              [ 0.  ,  0.  ,  0.  ,  1.  ]]),
        'CAM_FRONT_LEFT':np.array([[ 0.57357644,  0.81915204,  0.  , -0.22517331],
                                      [ 0.        ,  0.        , -1.  , -0.24      ],
                                   [-0.81915204,  0.57357644,  0.  , -0.82909407],
                                   [ 0.        ,  0.        ,  0.  ,  1.        ]]),
          'CAM_FRONT_RIGHT':np.array([[ 0.57357644, -0.81915204, 0.  ,  0.22517331],
                                   [ 0.        ,  0.        , -1.  , -0.24      ],
                                   [ 0.81915204,  0.57357644,  0.  , -0.82909407],
                                   [ 0.        ,  0.        ,  0.  ,  1.        ]]),
        'CAM_BACK':np.array([[-1. ,  0.,  0.,  0.  ],
                             [ 0. ,  0., -1., -0.24],
                             [ 0. , -1.,  0., -1.61],
                             [ 0. ,  0.,  0.,  1.  ]]),
     
        'CAM_BACK_LEFT':np.array([[-0.34202014,  0.93969262,  0.  , -0.25388956],
                                  [ 0.        ,  0.        , -1.  , -0.24      ],
                                  [-0.93969262, -0.34202014,  0.  , -0.49288953],
                                  [ 0.        ,  0.        ,  0.  ,  1.        ]]),
  
        'CAM_BACK_RIGHT':np.array([[-0.34202014, -0.93969262,  0.  ,  0.25388956],
                                  [ 0.        ,  0.         , -1.  , -0.24      ],
                                  [ 0.93969262, -0.34202014 ,  0.  , -0.49288953],
                                  [ 0.        ,  0.         ,  0.  ,  1.        ]])
        }
        self.lidar2ego = np.array([[ 0. ,  1. ,  0. , -0.39],
                                   [-1. ,  0. ,  0. ,  0.  ],
                                   [ 0. ,  0. ,  1. ,  1.84],
                                   [ 0. ,  0. ,  0. ,  1.  ]])
        self.vehicle_sensors =[
                # camera rgb
                {
                    'type': 'sensor.camera.rgb',
                    'x': 0.80, 'y': 0.0, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_FRONT'
                },
                {
                    'type': 'sensor.camera.rgb',
                    'x': 0.27, 'y': -0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -55.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_FRONT_LEFT'
                },
                {
                    'type': 'sensor.camera.rgb',
                    'x': 0.27, 'y': 0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 55.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_FRONT_RIGHT'
                },
                {
                    'type': 'sensor.camera.rgb',
                    'x': -2.0, 'y': 0.0, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 180.0,
                    'width': 1600, 'height': 900, 'fov': 110,
                    'id': 'CAM_BACK'
                },
                {
                    'type': 'sensor.camera.rgb',
                    'x': -0.32, 'y': -0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -110.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_BACK_LEFT'
                },
                {
                    'type': 'sensor.camera.rgb',
                    'x': -0.32, 'y': 0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 110.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_BACK_RIGHT'
                },
                # speed
                {
                    'type': 'sensor.speedometer',
                    'reading_frequency': 20,
                    'id': 'SPEED'
                },       
                # imu
                {
                    'type': 'sensor.other.imu',
                    'x': -1.4, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.05,
                    'id': 'IMU'
                },
                # gps
                {
                    'type': 'sensor.other.gnss',
                    'x': -1.4, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.01,
                    'id': 'GPS'
                },
                # bev
                {	
                    'type': 'sensor.camera.rgb',
                    'x': 0.0, 'y': 0.0, 'z': 50.0,
                    'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
                    'width': 512, 'height': 512, 'fov': 5 * 10.0,
                    'id': 'bev'
                }

            ]
        
        topdown_extrinsics =  np.array([[0.0, -0.0, -1.0, 50.0], [0.0, 1.0, -0.0, 0.0], [1.0, -0.0, 0.0, -0.0], [0.0, 0.0, 0.0, 1.0]])
        unreal2cam = np.array([[0,1,0,0], [0,0,-1,0], [1,0,0,0], [0,0,0,1]])
        self.coor2topdown = unreal2cam @ topdown_extrinsics
        topdown_intrinsics = np.array([[548.993771650447, 0.0, 256.0, 0], [0.0, 548.993771650447, 256.0, 0], [0.0, 0.0, 1.0, 0], [0, 0, 0, 1.0]])
        self.coor2topdown = topdown_intrinsics @ self.coor2topdown
        self.frame_rate = 20.0      # in Hz
        self.action_space_type = action_space_type
        self.action_type = action_type
        now = datetime.now()
        if self.action_type == 'plan':
            self.pidcontroller = PIDController() 
        
        
    def reset(self, _iter):
        self.traffic_manager_seed = 0
        self.statistics_manager = StatisticsManager(self.checkpoint,'./live_results.txt')
        self._init_carla_client()
        self._init_scenario_manager(self.statistics_manager)
        self.route_indexer = RouteIndexer(self.routes, self.repetitions, self.routes_subset)
        if self.resume:
            resume = self.route_indexer.validate_and_resume(self.checkpoint)
        else:
            resume = False

        if resume:
            self.statistics_manager.add_file_records(self.checkpoint)
        else:
            self.statistics_manager.clear_records()
        self.statistics_manager.save_progress(self.route_indexer.index, self.route_indexer.total)
        self.statistics_manager.write_statistics()
    def close(self):
        """清理资源，关闭环境"""
        if hasattr(self, 'route_scenario'):
            self._running = False  # 停止线程
            self.stop_scenario()
            for i, _ in enumerate(self._sensors_list):  # 停止传感器
                if self._sensors_list[i] is not None:
                    self._sensors_list[i].stop()
                    self._sensors_list[i].destroy()
                    self._sensors_list[i] = None
            self._sensors_list = []
            # Tick once to destroy the sensors
            CarlaDataProvider.get_world().tick()
            self._cleanup()
        self._reset_internal_state()

    def soft_reset(self, new_scene=True):
        self.close()
        self.SensorInterface = SensorInterface()
        self.config = self._load_new_scenario()
        if self.config is None:
            return None
        self._setup_world(self.config)

        if SAVE_PATH is not None:
            self.save_path = pathlib.Path(SAVE_PATH)/f"{self.config.name}_{self.config.repetition_index}"
            self.save_path.mkdir(parents=True, exist_ok=True)
            (self.save_path / 'rgb_front').mkdir(exist_ok=True)
            (self.save_path / 'rgb_front_right').mkdir(exist_ok=True)
            (self.save_path / 'rgb_front_left').mkdir(exist_ok=True)
            (self.save_path / 'rgb_back').mkdir(exist_ok=True)
            (self.save_path / 'rgb_back_right').mkdir(exist_ok=True)
            (self.save_path / 'rgb_back_left').mkdir(exist_ok=True)
            (self.save_path / 'meta').mkdir(exist_ok=True)
            (self.save_path / 'bev').mkdir(exist_ok=True)
        self._running = True
        self._start_run() 
        
        return True
    
    def _load_new_scenario(self):
        config = self.route_indexer.get_next_config()
        if config is None:
            print("All scence done")
            
        return config

    def _reset_internal_state(self):
        self.tick_count = 0
        self._timestamp_last_run = 0.0
        self.scenario_duration_system = 0.0
        self.scenario_duration_game = 0.0
        self.start_system_time = 0.0
        self.start_game_time = 0.0
        self.end_system_time = 0.0
        self.end_game_time = 0.0
        self._spectator = None
        self._watchdog = None
        self._agent_watchdog = None      

    def _init_carla_client(self, port=None):
        if port is None:
            port = self.port
        self.client = carla.Client(self.host, port)
        self.client.set_timeout(120.0)
        settings = carla.WorldSettings(
            synchronous_mode = True,
            fixed_delta_seconds = 1.0 / self.frame_rate, 
            deterministic_ragdolls = True,
            spectator_as_ego = False
        )
        try:
            self.client.get_world().apply_settings(settings)
        except Exception as e:
            print(f"An error occurred: {e}")
            
        print('\033[92m' + 'Build CARLA Client SUCCESS!!!!!!!!' + '\033[0m')

        self.traffic_manager = self.client.get_trafficmanager(self.traffic_manager_port)
        self.traffic_manager.set_synchronous_mode(True) 
        self.traffic_manager.set_hybrid_physics_mode(True) 
        self.traffic_manager.set_random_device_seed(self.traffic_manager_seed)
        print('\033[92m' + 'Build Traffic manager env SUCCESS!!!!!!!!' + '\033[0m')
    
    def _init_scenario_manager(self, statistics_manager):
        # Create the ScenarioManager
        self.manager = ScenarioManager(
            timeout=300, 
            statistics_manager = statistics_manager,
            debug_mode=self.debug
        )      
        print('\033[92m' + 'Build Scenario manager env SUCCESS!!!!!!!!' + '\033[0m')
    
    
    def _start_run(self):
        self.start_system_time = time.time()
        self.start_game_time = GameTime.get_time()

        # Detects if the simulation is down
        self._watchdog = Watchdog(self._timeout)
        self._watchdog.start()

        self._agent_watchdog = Watchdog(self._timeout)
        self._agent_watchdog.start()
        self._running = True

        # Thread for build_scenarios
        self._scenario_thread = threading.Thread(target=self.build_scenarios_loop, args=(self._debug_mode > 0, ))
        self._scenario_thread.start()


    def step(self, action_speed, action_path, _last_obs, command_speed=None, command_path= None,last_output = None,PENALTY_CONFIG=None):
        done = False
        self.crash_message = None
        reward= 0
        info = None
        input_data = None
        metadata_traj = {}
        results={}
        steer = 0.0
        throttle_traj = 0.0
        brake_traj = 0.0

        try:
            if action_speed is not None and action_path is not None:
                self._agent_watchdog.resume()
                self._agent_watchdog.update()
                
                steer, throttle_traj, brake_traj, metadata_traj = self.pidcontroller.control_pid(action_speed, action_path, _last_obs['can_bus'][0][7], command_speed)
                brake_traj = float(brake_traj)
                if brake_traj < 0.05: brake_traj = 0.0
                if throttle_traj > brake_traj: brake_traj = 0.0
        
                if _last_obs['can_bus'][0][7] >5:
                    throttle_traj = 0
                self.vehicle.control.steer = np.clip(float(steer), -1, 1)
                self.vehicle.control.throttle = np.clip(float(throttle_traj), 0, 0.75)
                self.vehicle.control.brake = np.clip(float(brake_traj), 0, 1)

                self._agent_watchdog.pause()
                self._watchdog.resume()
                self.vehicle.apply_control(self.vehicle.control)

                ego_trans = self.vehicle.get_transform()
                self.manager._spectator.set_transform(carla.Transform(ego_trans.location + carla.Location(z=70),
                                                        carla.Rotation(pitch=-90)))
                self._update_time()
                input_data = self.SensorInterface.get_data(GameTime.get_frame())
                self._print_simulation_status()
            else:
                self._update_time()
                input_data = self.SensorInterface.get_data(GameTime.get_frame())
                results, tick_data = self._prepare_data(input_data)
                self.pid_metadata = {}
                self.pid_metadata['agent'] = 'only_traj'
                self.pid_metadata['steer'] = self.vehicle.control.steer
                self.pid_metadata['throttle'] = self.vehicle.control.throttle
                self.pid_metadata['brake'] = self.vehicle.control.brake
                self.save(tick_data)
                self._print_simulation_status()
                return results, 0.0, False, {}
            py_trees.blackboard.Blackboard().set("AV_control", self.vehicle.control, overwrite=True)
            self.manager.scenario_tree.tick_once()# vehicle.get_carla_actor()
            if self.manager.scenario_tree.status != py_trees.common.Status.RUNNING:
                done = True
        except TickRuntimeError:
            done = True
            reward = 0
            self.entry_status, self.crash_message = "Started", "TickRuntime"
        except Exception as e:
            print(f"Error in step function: {e}")
            done = True
            reward = -1
            self.crash_message = f"Exception: {str(e)}"

        if input_data is not None:
            results, tick_data = self._prepare_data(input_data)

        self.pid_metadata = metadata_traj
        self.pid_metadata['agent'] = 'only_traj'
        self.pid_metadata['steer'] = self.vehicle.control.steer
        self.pid_metadata['throttle'] = self.vehicle.control.throttle
        self.pid_metadata['brake'] = self.vehicle.control.brake
        self.pid_metadata['steer_traj'] = float(steer)
        self.pid_metadata['throttle_traj'] = float(throttle_traj)
        self.pid_metadata['brake_traj'] = float(brake_traj)
        if action_path is not None:
             self.pid_metadata['plan'] = action_path.tolist()
        self.pid_metadata['command_speed'] = command_speed

        for node in self.route_scenario.get_criteria():
            for event in node.events:
                # print(event.get_type().name)
                if PENALTY_CONFIG is not None:
                    if event.get_type().name in PENALTY_CONFIG:
                        crash_message = event.get_type().name 
                        done = True
                        reward = -1
                        self.entry_status, self.crash_message = "Started", crash_message
                    elif event.get_type().name == 'ROUTE_COMPLETION':
                        score_route = event.get_dict()['route_completed']
                        target_reached = score_route >= 100
                        if target_reached:
                            done = True
                            reward = 1
                            self.entry_status, self.crash_message = "Started", "SUCCESS"
 
        # save
        if SAVE_PATH is not None and self.tick_count % 10 == 0:
            if last_output is not None and len(last_output) > 0:
                try:
                    result_vis = last_output[0]['pts_bbox'].get('result_vis', None)
                    value = last_output[0]['pts_bbox']['ppo_info']['values']
                    self.save(tick_data, result_vis, value=value)
                except Exception as e:
                    print(f"Error saving data: {e}")
            else:
                self.save(tick_data)

        return results, reward, done, info
    
    def _update_time(self):
        if self._running and self.get_running_status():
            CarlaDataProvider.get_world().tick(self._timeout)
        timestamp = CarlaDataProvider.get_world().get_snapshot().timestamp
        if self._timestamp_last_run < timestamp.elapsed_seconds:
            self._timestamp_last_run = timestamp.elapsed_seconds
            self._watchdog.update()
            GameTime.on_carla_tick(timestamp)
            CarlaDataProvider.on_carla_tick()
            self.tick_count += 1
            self._watchdog.pause()

            if self.tick_count > self.max_tick_count:
                raise TickRuntimeError("tick_count > 3000")

    def _print_simulation_status(self): 
        timestamp = GameTime.get_time()
        if not self.wallclock_t0:
            self.wallclock_t0 = GameTime.get_wallclocktime()
        wallclock = GameTime.get_wallclocktime()
        wallclock_diff = (wallclock - self.wallclock_t0).total_seconds()
        sim_ratio = 0 if wallclock_diff == 0 else timestamp/wallclock_diff
        
        
        print('=== [Agent] -- Wallclock = {} -- System time = {} -- Game time = {} -- Ratio = {}x'.format(
            str(wallclock)[:-3], 
            format(wallclock_diff, '.3f'), 
            format(timestamp, '.3f'), 
            format(sim_ratio, '.3f')), flush=True)

    def _compute_reward(self):
        route_progress = self._statistics_manager.get_route_progress()
        collision_penalty = -10.0 if self._collision_detected else 0.0
        return route_progress + collision_penalty
    
    def _set_global_plan(self, global_plan_gps, global_plan_world_coord):
        ds_ids = downsample_route(global_plan_world_coord, 1)
        self._global_plan_world_coord = [(global_plan_world_coord[x][0], global_plan_world_coord[x][1]) for x in ds_ids]
        self._global_plan = [global_plan_gps[x] for x in ds_ids]
        self._plan_gps_HACK = global_plan_gps

        locx, locy = self._global_plan_world_coord[0][0].location.x, self._global_plan_world_coord[0][0].location.y
        lon, lat = self._global_plan[0][0]['lon'], self._global_plan[0][0]['lat']
        EARTH_RADIUS_EQUA = 6378137.0
        def equations(vars):
            x, y = vars
            eq1 = lon * math.cos(x * math.pi / 180) - (locx * x * 180) / (math.pi * EARTH_RADIUS_EQUA) - math.cos(x * math.pi / 180) * y
            eq2 = math.log(math.tan((lat + 90) * math.pi / 360)) * EARTH_RADIUS_EQUA * math.cos(x * math.pi / 180) + locy - math.cos(x * math.pi / 180) * EARTH_RADIUS_EQUA * math.log(math.tan((90 + x) * math.pi / 360))
            return [eq1, eq2]
        initial_guess = [0, 0]
        solution = fsolve(equations, initial_guess)
        self.lat_ref, self.lon_ref = solution[0], solution[1]
        self._route_planner = RoutePlanner(4.0, 50.0, lat_ref=self.lat_ref, lon_ref=self.lon_ref)
        self._route_planner.set_route(global_plan_gps, True)

    def _prepare_data(self,input_data):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 20]
        imgs = {}
        for cam in ['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT']:
            # img = cv2.cvtColor(input_data[cam][1][:, :, :3], cv2.COLOR_BGR2RGB)
            img = input_data[cam][1][:, :, :3]
            _, img = cv2.imencode('.jpg', img, encode_param)
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            imgs[cam] = img
            # cv2.imwrite(f'./carla/{cam}_{self.tick_count}.jpg', img)
        # cv2.imwrite('./work_dirs/tick_input_img.jpg', img)
        # bev = cv2.cvtColor(input_data['bev'][1][:, :, :3], cv2.COLOR_BGR2RGB)

        bev = input_data['bev'][1][:, :, :3]
        gps = input_data['GPS'][1][:2]
        speed = input_data['SPEED'][1]['speed']
        compass = input_data['IMU'][1][-1]
        acceleration = input_data['IMU'][1][:3]
        angular_velocity = input_data['IMU'][1][3:6]

        pos = self.gps_to_location(gps)
        (_, curr_command), (near_node, near_command) = self._route_planner.run_step(pos)

        if (math.isnan(compass) == True): #It can happen that the compass sends nan for a few frames
            compass = 0.0
            acceleration = np.zeros(3)
            angular_velocity = np.zeros(3)

        tick_data = {
                'imgs': imgs,
                'gps': gps,
                'pos':pos,
                'speed': speed,
                'compass': compass,
                'bev': bev,
                'acceleration':acceleration,
                'angular_velocity':angular_velocity,
                'command_curr':curr_command,
                'command_near':near_command,
                'command_near_xy':near_node
                }
        results = {}
        results['lidar2img'] = []
        results['lidar2cam'] = []
        results['cam_intrinsic'] = []
        results['img'] = []
        results['folder'] = ' '
        results['scene_token'] = ' '  
        results['frame_idx'] = self.tick_count
        results['timestamp'] = self.tick_count / 20
        # results['box_type_3d'], _ = get_box_type('LiDAR')
  
        for cam in ['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT']:
            results['lidar2img'].append(self.lidar2img[cam])
            results['lidar2cam'].append(self.lidar2cam[cam])
            results['cam_intrinsic'].append(np.matmul(self.lidar2img[cam], np.linalg.inv(self.lidar2cam[cam])))
            results['img'].append(tick_data['imgs'][cam])
        results['lidar2img'] = np.stack(results['lidar2img'],axis=0)
        results['lidar2cam'] = np.stack(results['lidar2cam'],axis=0)
        raw_theta = tick_data['compass']   if not np.isnan(tick_data['compass']) else 0
        ego_theta = -raw_theta + np.pi/2
        rotation = list(Quaternion(axis=[0, 0, 1], radians=ego_theta))
        can_bus = np.zeros(18)
        can_bus[0] = tick_data['pos'][0]
        can_bus[1] = -tick_data['pos'][1]
        can_bus[3:7] = rotation
        can_bus[7] = tick_data['speed']
        can_bus[10:13] = tick_data['acceleration']
        can_bus[11] *= -1
        can_bus[13:16] = -tick_data['angular_velocity']
        can_bus[16] = ego_theta
        can_bus[17] = ego_theta / np.pi * 180 
        results['can_bus'] = can_bus
        command = tick_data['command_curr']
        results['command'] = command2nohot(tick_data['command_curr'])
        results['ego_fut_cmd'] = command2hot(tick_data['command_curr'])
  
        theta_to_lidar = raw_theta
        command_near_xy = np.array([tick_data['command_near_xy'][0]-can_bus[0],-tick_data['command_near_xy'][1]-can_bus[1]])
        rotation_matrix = np.array([[np.cos(theta_to_lidar),-np.sin(theta_to_lidar)],[np.sin(theta_to_lidar),np.cos(theta_to_lidar)]])
        local_command_xy = rotation_matrix @ command_near_xy
  
        ego2world = np.eye(4)
        ego2world[0:3,0:3] = Quaternion(axis=[0, 0, 1], radians=ego_theta).rotation_matrix
        ego2world[0:2,3] = can_bus[0:2]
        ego_pose = ego2world
        ego_pose_inv = invert_matrix_egopose_numpy(ego_pose)
        results['ego_pose'] = ego_pose
        results['ego_pose_inv'] = ego_pose_inv
        lidar2global = ego2world @ self.lidar2ego
        ego_pose = lidar2global
        ego_pose_inv = invert_matrix_egopose_numpy(ego_pose)
        results['ego_pose'] = ego_pose
        results['ego_pose_inv'] = ego_pose_inv
        results['lidar2ego'] = self.lidar2ego
        results['l2g_r_mat'] = lidar2global[0:3,0:3]
        results['l2g_t'] = lidar2global[0:3,3]
        stacked_imgs = np.stack(results['img'],axis=-1)
        results['img_shape'] = stacked_imgs.shape
        results['ori_shape'] = stacked_imgs.shape
        results['pad_shape'] = stacked_imgs.shape
        return results, tick_data


    def save(self, tick_data, result_vis=None,value=None):

        frame = self.tick_count//10
        cvt_c = lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # PIL save uses RGB image 
        # Image.fromarray(cvt_c(tick_data['imgs']['CAM_FRONT'])).save(self.save_path / 'rgb_front' / ('%04d.png' % frame))
        # Image.fromarray(cvt_c(tick_data['imgs']['CAM_FRONT_LEFT'])).save(self.save_path / 'rgb_front_left' / ('%04d.png' % frame))
        # Image.fromarray(cvt_c(tick_data['imgs']['CAM_FRONT_RIGHT'])).save(self.save_path / 'rgb_front_right' / ('%04d.png' % frame))
        # Image.fromarray(cvt_c(tick_data['imgs']['CAM_BACK'])).save(self.save_path / 'rgb_back' / ('%04d.png' % frame))
        # Image.fromarray(cvt_c(tick_data['imgs']['CAM_BACK_LEFT'])).save(self.save_path / 'rgb_back_left' / ('%04d.png' % frame))
        # Image.fromarray(cvt_c(tick_data['imgs']['CAM_BACK_RIGHT'])).save(self.save_path / 'rgb_back_right' / ('%04d.png' % frame))
        # Image.fromarray(cvt_c(tick_data['bev'])).save(self.save_path / 'bev' / ('%04d.png' % frame))
        # outfile = open(self.save_path / 'meta' / ('%04d.json' % frame), 'w')
        # json.dump(self.pid_metadata, outfile, indent=4)
        # outfile.close()
        
        if result_vis is not None:
            img_to_show = result_vis['img_to_show']
            img_bev = result_vis['img_bev']

            meta = self.pid_metadata
            def put_text(img, text, pos):
                return cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1, 1, cv2.LINE_AA)
            meta_text = f"speed: {meta['speed']:.3f}" 
            img_bev = put_text(img_bev, meta_text, (20, 100))
            meta_text = f"steer: {meta['steer']:.3f}" 
            img_bev = put_text(img_bev, meta_text, (20, 120))
            meta_text = f"throttle: {meta['throttle']:.3f}" 
            img_bev = put_text(img_bev, meta_text, (20, 140))
            meta_text = f"brake: {meta['brake']:.3f}" 
            img_bev = put_text(img_bev, meta_text, (20, 160))
            if value is not None:
                meta_text = f"value: {value.item():.3f}" 
                img_bev = put_text(img_bev, meta_text, (20, 180))


            img = np.concatenate([img_to_show, img_bev], axis=1)
            if 'qa_img' in result_vis and not np.all(result_vis['qa_img']==255):
                 img = np.concatenate([img, result_vis['qa_img']], axis=0)
            save_dir = os.path.join(self.save_path, 'vis')
            os.makedirs(save_dir, exist_ok=True)
            cv2.imwrite(f"{save_dir}/{'%04d.png' % frame}", img)
    
    def _setup_world(self, config):
        max_attempts = 3 
        attempt= 0
        # config = self.route_indexer.get_next_config()
        crash_message = ""
        self.entry_status = "Started"
        print("\n\033[1m========= Preparing {} (repetition {}) =========\033[0m".format(config.name, config.repetition_index), flush=True)
        # Prepare the statistics of the route
        route_name = f"{config.name}_rep{config.repetition_index}"
        scenario_name = config.scenario_configs[0].name
        town_name = str(config.town)
        weather_id = self.get_weather_id(config.weather[0][1])
        currentDateAndTime = datetime.now()
        currentTime = currentDateAndTime.strftime("%m_%d_%H_%M_%S")
        save_name = f"{route_name}_{town_name}_{scenario_name}_{weather_id}_{currentTime}"
        self.statistics_manager.create_route_data(route_name, scenario_name, weather_id, save_name, town_name, config.index)
        
        print("\033[1m> Loading the world\033[0m", flush=True)

        # Load the world and the scenario
        try:
            # with CarlaDataProvider._lock:
            while attempt < max_attempts:
                try:
                    self._load_and_wait_for_world(config.town)
                    break
                except Exception as e:
                    print(f"发生错误: {e}")
                
            if CarlaDataProvider._world is None or CarlaDataProvider._map is None:
                raise ValueError("World or Map has not been initialized correctly!")
            self.route_scenario = RouteScenario(world=CarlaDataProvider.get_world(), config=config, debug_mode=self.debug)
            
            self.route = self.route_scenario.route
            self.route_gps = self.route_scenario.gps_route
            self._set_global_plan(self.route_gps, self.route)
            self.statistics_manager.set_scenario(self.route_scenario)
            
        except Exception:
            print("\n\033[91mThe scenario could not be loaded:", flush=True)
            print(f"\n{traceback.format_exc()}\033[0m", flush=True)

            self.entry_status, crash_message = FAILURE_MESSAGES["Simulation"]
            self._register_statistics(config.index, self.entry_status, crash_message)
        
        print("\033[1m> Setting up the agent and route\033[0m", flush=True)
        ego_actor = self.route_scenario.ego_vehicles
        self.vehicle = My_Vehicle(
            CarlaDataProvider.get_world(),
            ego_actor[0],
            on_collision_fn=None,
            on_invasion_fn=None, #
            is_ego=True
        )
        self.manager.my_load_scenario(self.route_scenario, self.vehicle, config.index, config.repetition_index)
        
        self.setup_sensors(self.vehicle) # 初始化传感器
        print('\033[92m' + 'BUILD SENSOR CAMS, IMU, GPS SUCCESS!!!!!!!!' + '\033[0m')
        print("\033[1m> Running the route\033[0m", flush=True)

        self.manager.tick_count = 0
        self.route_index = config.index
    
    def build_scenarios_loop(self, debug):
        """
        Keep periodically trying to start the scenarios that are close to the ego vehicle
        Additionally, do the same for the spawned vehicles
        """
        while self._running:
            self.route_scenario.build_scenarios(self.vehicle, debug=debug)
            self.route_scenario.spawn_parked_vehicles(self.vehicle)
            time.sleep(1)
        
    def _load_and_wait_for_world(self, town):
        """CarlaDataProvider.get_location(self.vehicle)
        Load a new CARLA world without changing the settings and provide data to CarlaDataProvider
        """
        self.world = self.client.load_world(town)

        # Large Map settings are always reset, for some reason
        settings = self.world.get_settings()
        settings.tile_stream_distance = 650
        settings.actor_active_distance = 650
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / self.frame_rate

        self.world.apply_settings(settings)

        self.world.reset_all_traffic_lights()

        CarlaDataProvider.set_client(self.client)
        CarlaDataProvider.set_traffic_manager_port(self.traffic_manager_port)
        CarlaDataProvider.set_world(self.world)

        # This must be here so that all route repetitions use the same 'unmodified' seed
        self.traffic_manager.set_random_device_seed(self.traffic_manager_seed)

        # Wait for the world to be ready
        self.world.tick()

        map_name = CarlaDataProvider.get_map().name.split("/")[-1]
        if map_name != town:
            raise Exception("The CARLA server uses the wrong map!"
                            " This scenario requires the use of map {}".format(town))
        return True
    def gps_to_location(self, gps):
        EARTH_RADIUS_EQUA = 6378137.0
        # gps content: numpy array: [lat, lon, alt]
        lat, lon = gps
        scale = math.cos(self.lat_ref * math.pi / 180.0)
        my = math.log(math.tan((lat+90) * math.pi / 360.0)) * (EARTH_RADIUS_EQUA * scale)
        mx = (lon * (math.pi * EARTH_RADIUS_EQUA * scale)) / 180.0
        y = scale * EARTH_RADIUS_EQUA * math.log(math.tan((90.0 + self.lat_ref) * math.pi / 360.0)) - my
        x = mx - scale * self.lon_ref * math.pi * EARTH_RADIUS_EQUA / 180.0
        return np.array([x, y])

    def get_running_status(self):
        """
        returns:
           bool: False if watchdog exception occured, True otherwise
        """
        if self._watchdog:
            return self._watchdog.get_status()
        return True
    
    def get_weather_id(self, weather_conditions):
        from xml.etree import ElementTree as ET
        tree = ET.parse('./data/weather.xml')
        root = tree.getroot()
        def conditions_match(weather, conditions):
            for (key, value) in weather:
                if key == 'route_percentage' : continue
                if str(getattr(conditions, key))!= value:
                    return False
            return True
        for case in root.findall('case'):
            weather = case[0].items()
            if conditions_match(weather, weather_conditions):
                return case.items()[0][1]
        return None
    
    def _set_route_planner(self, global_plan_gps, global_plan_world):
        locx, locy = global_plan_world[0][0].location.x, global_plan_world[0][0].location.y
        lon, lat = global_plan_gps[0][0]['lon'], global_plan_gps[0][0]['lat']
        EARTH_RADIUS_EQUA = 6378137.0
        def equations(vars):
            x, y = vars
            eq1 = lon * math.cos(x * math.pi / 180) - (locx * x * 180) / (math.pi * EARTH_RADIUS_EQUA) - math.cos(x * math.pi / 180) * y
            eq2 = math.log(math.tan((lat + 90) * math.pi / 360)) * EARTH_RADIUS_EQUA * math.cos(x * math.pi / 180) + locy - math.cos(x * math.pi / 180) * EARTH_RADIUS_EQUA * math.log(math.tan((90 + x) * math.pi / 360))
            return [eq1, eq2]
        initial_guess = [0, 0]
        solution = fsolve(equations, initial_guess)
        self.lat_ref, self.lon_ref = solution[0], solution[1]
        self._route_planner = RoutePlanner(4.0, 50.0, lat_ref=self.lat_ref, lon_ref=self.lon_ref)
        self._route_planner.set_route(global_plan_gps, True)

    def _preprocess_sensor_spec(self, sensor_spec):
        type_ = sensor_spec["type"]
        id_ = sensor_spec["id"]
        attributes = {}

        if type_ == 'sensor.opendrive_map':
            attributes['reading_frequency'] = sensor_spec['reading_frequency']
            sensor_location = carla.Location()
            sensor_rotation = carla.Rotation()

        elif type_ == 'sensor.speedometer':
            delta_time = CarlaDataProvider.get_world().get_settings().fixed_delta_seconds
            attributes['reading_frequency'] = 1 / delta_time
            sensor_location = carla.Location()
            sensor_rotation = carla.Rotation()

        if type_ == 'sensor.camera.rgb':
            attributes['image_size_x'] = str(sensor_spec['width'])
            attributes['image_size_y'] = str(sensor_spec['height'])
            attributes['fov'] = str(sensor_spec['fov'])

            sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'],
                                             z=sensor_spec['z'])
            sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                             roll=sensor_spec['roll'],
                                             yaw=sensor_spec['yaw'])

        elif type_ == 'sensor.lidar.ray_cast':
            attributes['range'] = str(85)
            attributes['rotation_frequency'] = str(10)
            attributes['channels'] = str(64)
            attributes['upper_fov'] = str(10)
            attributes['lower_fov'] = str(-30)
            attributes['points_per_second'] = str(600000)
            attributes['atmosphere_attenuation_rate'] = str(0.004)
            attributes['dropoff_general_rate'] = str(0.45)
            attributes['dropoff_intensity_limit'] = str(0.8)
            attributes['dropoff_zero_intensity'] = str(0.4)

            sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'],
                                             z=sensor_spec['z'])
            sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                             roll=sensor_spec['roll'],
                                             yaw=sensor_spec['yaw'])

        elif type_ == 'sensor.other.radar':
            attributes['horizontal_fov'] = str(sensor_spec['horizontal_fov'])  # degrees
            attributes['vertical_fov'] = str(sensor_spec['vertical_fov'])  # degrees
            attributes['points_per_second'] = '1500'
            attributes['range'] = '100'  # meters

            sensor_location = carla.Location(x=sensor_spec['x'],
                                             y=sensor_spec['y'],
                                             z=sensor_spec['z'])
            sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                             roll=sensor_spec['roll'],
                                             yaw=sensor_spec['yaw'])

        elif type_ == 'sensor.other.gnss':
            attributes['noise_alt_stddev'] = str(0.000005)
            attributes['noise_lat_stddev'] = str(0.000005)
            attributes['noise_lon_stddev'] = str(0.000005)
            attributes['noise_alt_bias'] = str(0.0)
            attributes['noise_lat_bias'] = str(0.0)
            attributes['noise_lon_bias'] = str(0.0)

            sensor_location = carla.Location(x=sensor_spec['x'],
                                             y=sensor_spec['y'],
                                             z=sensor_spec['z'])
            sensor_rotation = carla.Rotation()

        elif type_ == 'sensor.other.imu':
            attributes['noise_accel_stddev_x'] = str(0.001)
            attributes['noise_accel_stddev_y'] = str(0.001)
            attributes['noise_accel_stddev_z'] = str(0.015)
            attributes['noise_gyro_stddev_x'] = str(0.001)
            attributes['noise_gyro_stddev_y'] = str(0.001)
            attributes['noise_gyro_stddev_z'] = str(0.001)

            sensor_location = carla.Location(x=sensor_spec['x'],
                                             y=sensor_spec['y'],
                                             z=sensor_spec['z'])
            sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                             roll=sensor_spec['roll'],
                                             yaw=sensor_spec['yaw'])
        sensor_transform = carla.Transform(sensor_location, sensor_rotation)

        return type_, id_, sensor_transform, attributes

    def stop_scenario(self):
        """
        This function triggers a proper termination of a scenario
        """
        if self._watchdog:
            self._watchdog.stop()

        if self._agent_watchdog:
            self._agent_watchdog.stop()

        self.compute_duration_time()

        if self.get_running_status():
            if self.route_scenario is not None:
                self.route_scenario.terminate()

            self.analyze_scenario()

        # Make sure the scenario thread finishes to avoid blocks
        self._running = False
        if self._scenario_thread is not None:
            self._scenario_thread.join() 
        self._scenario_thread = None

    def analyze_scenario(self):
        """
        Analyzes and prints the results of the route
        """
        ResultOutputProvider(self)

    def compute_duration_time(self):
        """
        Computes system and game duration times
        """
        self.end_system_time = time.time()
        self.end_game_time = GameTime.get_time()

        self.scenario_duration_system = self.end_system_time - self.start_system_time
        self.scenario_duration_game = self.end_game_time - self.start_game_time

    def setup_sensors(self, vehicle):
        """
        Create the sensors defined by the user and attach them to the ego-vehicle
        :param vehicle: ego vehicle
        :return:
        """
        world = CarlaDataProvider.get_world()
        bp_library = world.get_blueprint_library()
        for sensor_spec in self.vehicle_sensors:
            type_, id_, sensor_transform, attributes = self._preprocess_sensor_spec(sensor_spec)
            
            if type_ == 'sensor.speedometer':
                sensor = SpeedometerReader(vehicle.get_carla_actor(), attributes['reading_frequency'])
            else:
                bp = bp_library.find(type_)
                for key, value in attributes.items():
                    bp.set_attribute(str(key), str(value))
                sensor = CarlaDataProvider.get_world().spawn_actor(bp, sensor_transform, vehicle.get_carla_actor())

            # setup callback
            sensor.listen(CallBack(id_, type_, sensor, self.SensorInterface))
            self._sensors_list.append(sensor)
            # print(sensor.intrinsics)

        # Some sensors miss sending data during the first ticks, so tick several times and remove the data
        for _ in range(10):
            world.tick()


    def _register_statistics(self, route_index, entry_status, crash_message=""):
        """
        Computes and saves the route statistics
        """
        print("\033[1m> Registering the route statistics\033[0m", flush=True)
        self.statistics_manager.save_entry_status(entry_status)
        self.statistics_manager.compute_route_statistics(
            route_index, self.manager.scenario_duration_system, self.manager.scenario_duration_game, crash_message
        )
        
    def _on_collision(self, event):
        if get_actor_display_name(event.other_actor) != "Road":
            self.terminal_state = True
            self.collision_state = True
            # print("0| Terminal:  Collision with {}".format(event.other_actor.type_id))
            logger.warning("Terminal : Collision with {}".format(event.other_actor.type_id))

        # if self.activate_render:
        #     self.hud.notification("Collision with {}".format(get_actor_display_name(event.other_actor)))
    def _cleanup(self):
        """
        Remove and destroy all actors
        """
        CarlaDataProvider.cleanup()

        if self._agent_watchdog:
            self._agent_watchdog.stop()

        # try:
        #     if self.agent_instance:
        #         self.agent_instance.destroy()
        #         self.agent_instance = None
        # except Exception as e:
        #     print("\n\033[91mFailed to stop the agent:", flush=True)
        #     print(f"\n{traceback.format_exc()}\033[0m", flush=True)

        if self.route_scenario:
            self.route_scenario.remove_all_actors()
            self.route_scenario = None
            if self.statistics_manager:
                self.statistics_manager.remove_scenario()

        if self.manager:
            self._client_timed_out = not self.manager.get_running_status()
            self.manager.cleanup()

        # Make sure no sensors are left streaming
        alive_sensors = self.world.get_actors().filter('*sensor*')
        for sensor in alive_sensors:
            sensor.stop()
            sensor.destroy()


def command2hot(command,max_dim=6):
    if command < 0:
        command = 4
    command -= 1
    cmd_one_hot = np.zeros(max_dim)
    cmd_one_hot[command] = 1
    return cmd_one_hot

def command2nohot(command,max_dim=6):
    if command < 0:
        command = 4
    command -= 1
    return command

def invert_matrix_egopose_numpy(egopose):
    """ Compute the inverse transformation of a 4x4 egopose numpy matrix."""
    inverse_matrix = np.zeros((4, 4), dtype=np.float32)
    rotation = egopose[:3, :3]
    translation = egopose[:3, 3]
    inverse_matrix[:3, :3] = rotation.T
    inverse_matrix[:3, 3] = -np.dot(rotation.T, translation)
    inverse_matrix[3, 3] = 1.0
    return inverse_matrix