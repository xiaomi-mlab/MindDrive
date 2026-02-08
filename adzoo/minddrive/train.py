# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
 
from __future__ import division

import argparse
import copy
import mmcv
import os
import time
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.utils import get_dist_info, init_dist
from os import path as osp


from mmcv.datasets import build_dataset
from mmcv.models import build_model
from mmcv.utils import collect_env, get_root_logger
from mmcv.utils import set_random_seed

from mmcv.utils import TORCH_VERSION, digit_version
from adzoo.minddrive.apis.train import custom_train_model

import socket
import cv2
cv2.setNumThreads(1)

import sys
sys.path.append('')

import subprocess
def print_trainable_and_untrainable_params(model):
    """
    打印模型中所有训练的参数模型名称和不训练的参数模型名称。
    
    参数:
        model: PyTorch模型实例
    """
    trainable_params = []
    untrainable_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(name)
        else:
            untrainable_params.append(name)

    print("训练的参数模型名称:")
    for param_name in trainable_params:
        print(param_name)

    print("\n不训练的参数模型名称:")
    for param_name in untrainable_params:
        print(param_name)

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--use_carla', 
        action='store_true',
        help='use carla to collect')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
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
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--load_from', 
        type=str, 
        default=None, 
        help='the checkpoint file to load from'
    )
    parser.add_argument(
        '--rollout_data', 
        type=str, 
        default=None, 
        help='the rollout data file to train'
    )
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both specified, '
            '--options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # set tf32
    if cfg.get('close_tf32', False):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    # if args.resume_from is not None:
    if args.resume_from is not None and osp.isfile(args.resume_from):
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)
    if digit_version(TORCH_VERSION) == digit_version('1.8.1') and cfg.optimizer['type'] == 'AdamW':
        cfg.optimizer['type'] = 'AdamW2' # fix bug in Adamw
    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    # specify logger name, if we still use 'mmdet', the output info will be
    # filtered and won't be saved in the log_file
    # TODO: ugly workaround to judge whether we are training det or seg model
    if cfg.model.type in ['EncoderDecoder3D']:
        logger_name = 'mmseg'
    else:
        logger_name = 'mmdet'
    logger = get_root_logger(
        log_file=log_file, log_level=cfg.log_level, name=logger_name)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if args.seed is not None:
        if not args.deterministic:
            seed = int(time.time())
            set_random_seed(seed, deterministic=args.deterministic)
        else:
            seed=args.seed
            set_random_seed(seed, deterministic=args.deterministic)
    logger.info(f'Set random seed to {seed}, '
        f'deterministic: {args.deterministic}')
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)

    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    
    is_rl_training = cfg.get('rl_training', False)

    if is_rl_training:
        logger.info("RL Training mode. Adjusting requires_grad for specific layers...")
        for name, param in model.named_parameters():
            param.requires_grad = False
            if "lora_" in name and 'decision_expert' in name:
                param.requires_grad = True
            elif "value_net_pro" in name:
                param.requires_grad = True
        trainable_params = [n for n, p in model.named_parameters() if p.requires_grad]
        logger.info(f"Trainable parameters count: {len(trainable_params)}")


    logger.info(f'Model:\n{model}')
    print_trainable_and_untrainable_params(model)    
    datasets = []

    if args.rollout_data is not None:
        cfg.data.rollout.data_root = args.rollout_data
    for _, (mode, iters) in enumerate(cfg.workflow):
        if mode == 'train':
            dataset = build_dataset(cfg.data.train)
        elif mode == 'rollout' or mode == 'ppo_train':
            dataset = build_dataset(cfg.data.rollout)
        elif mode == 'val':
            val_dataset = copy.deepcopy(cfg.data.val)
            if 'dataset' in cfg.data.train:
                val_dataset.pipeline = cfg.data.train.dataset.pipeline
            else:
                val_dataset.pipeline = cfg.data.train.pipeline
            val_dataset.test_mode = False
            dataset = build_dataset(val_dataset)
        datasets.append(dataset)
    
    if cfg.checkpoint_config is not None:
        cfg.checkpoint_config.meta = dict(
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE  # for segmentors
            if hasattr(datasets[0], 'PALETTE') else None)
    
    model.CLASSES = datasets[0].CLASSES
    custom_train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=False,
        timestamp=timestamp,
        meta=meta,
        args=args)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('fork', force=True) 
    main()
