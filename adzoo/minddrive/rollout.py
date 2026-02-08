# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
 
from __future__ import division

import argparse
import os
import time
import torch
from mmcv import Config
from mmcv.models import build_model
from mmcv.utils import set_random_seed
from adzoo.minddrive.apis.rollout import custom_collect_dataset
import socket

def find_free_port(starting_port):
    port = starting_port
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("localhost", port))
                return port
        except OSError:
            port += 1
import cv2
cv2.setNumThreads(1)

import sys
sys.path.append('')

import subprocess

def parse_args():
    parser = argparse.ArgumentParser(description='Collect a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    # for rollout
    parser.add_argument(
        '--use_carla', action='store_true',help='use carla to collect')
    parser.add_argument('--host', default='localhost',
                help='IP of the host server (default: localhost)')
    parser.add_argument('--port', default=2000, type=int,
                    help='TCP port to listen to (default: 2000)')   
    parser.add_argument('--traffic_manager_port', default=8000, type=int,
                        help='Port to use for the TrafficManager (default: 8000)')
    parser.add_argument('--traffic-manager-seed', default=0, type=int,
                        help='Seed used by the TrafficManager (default: 0)')
    parser.add_argument("--checkpoint", type=str, default='./simulation_results.json',
                        help="Path to checkpoint used for saving statistics and resuming")                
    parser.add_argument('--routes', type=str, default='',
                        help='Name of the routes file to be executed.')
    parser.add_argument('--repetitions', type=int, default=1,
                        help='Number of repetitions per route.')
    parser.add_argument('--resume', action='store_true',
                        help='Resume execution from last checkpoint?')
    parser.add_argument("--gpu-rank", type=int, default=0)
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)
    # set random seeds
    if args.seed is not None:
        if not args.deterministic:
            seed = int(time.time())
            set_random_seed(seed, deterministic=args.deterministic)
        else:
            seed=args.seed
            set_random_seed(seed, deterministic=args.deterministic)
    # build model from config
    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
        
    model.cuda()
    model.eval()
    print(f'Model:\n{model}')

    if args.use_carla:
        # carla server start
        CARLA_ROOT = os.environ["CARLA_ROOT"]
        carla_path = os.path.join(CARLA_ROOT, "CarlaUE4.sh")
        args.port = find_free_port(args.port)
        port = args.port
        launch_command = [carla_path]
        launch_command += ['-RenderOffScreen']
        launch_command += ['-nosound']
        launch_command += [f'-carla-world-port={port}']
        launch_command += [f'-graphicsadapter=0']
        launch_command = " ".join(launch_command)
        launch_command = f"su - carla -c \"{launch_command}\""
        print("Running command:")
        print("".join(launch_command))
        carla_process = subprocess.Popen(launch_command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        print("Waiting for CARLA to initialize\n")
        time.sleep(10)
        if carla_process.poll() is None:
            print("CARLA process is running")
        else:
            print("CARLA process has exited with code", carla_process.poll())
    
    custom_collect_dataset(model, cfg=cfg, args=args)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('fork', force=True) 
    main()
