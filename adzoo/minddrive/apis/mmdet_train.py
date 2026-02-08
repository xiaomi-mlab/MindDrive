# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
import random
import warnings

import numpy as np
import torch
import torch.distributed as dist
from torch.nn import DataParallel
from torch.nn.parallel.distributed import DistributedDataParallel
from mmcv.runner import (HOOKS, DistSamplerSeedHook, EpochBasedRunner,
                         Fp16OptimizerHook, OptimizerHook,
                         build_runner, )
from mmcv.optims import build_optimizer
from mmcv.utils import build_from_cfg

from mmcv.core import EvalHook

from mmcv.datasets import (build_dataset, replace_ImageToTensor)
from mmcv.utils import get_root_logger, get_dist_info
import time
import os.path as osp
from mmcv.datasets import build_dataloader
from mmcv.core.evaluation.eval_hooks import CustomDistEvalHook
from adzoo.minddrive.apis.test import custom_multi_gpu_test
from mmcv.datasets.pipelines import Compose


    
def custom_train_detector(model,
                   dataset,
                   cfg,
                   distributed=False,
                   validate=False,
                   timestamp=None,
                   eval_model=None,
                   meta=None,
                   args=None,
                   ):
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
   
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    if 'imgs_per_gpu' in cfg.data:
        logger.warning('"imgs_per_gpu" is deprecated in MMDet V2.0. '
                       'Please use "samples_per_gpu" instead')
        if 'samples_per_gpu' in cfg.data:
            logger.warning(
                f'Got "imgs_per_gpu"={cfg.data.imgs_per_gpu} and '
                f'"samples_per_gpu"={cfg.data.samples_per_gpu}, "imgs_per_gpu"'
                f'={cfg.data.imgs_per_gpu} is used in this experiments')
        else:
            logger.warning(
                'Automatically set "samples_per_gpu"="imgs_per_gpu"='
                f'{cfg.data.imgs_per_gpu} in this experiments')
        cfg.data.samples_per_gpu = cfg.data.imgs_per_gpu

    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
            shuffle = False,
            shuffler_sampler=cfg.data.shuffler_sampler,  # dict(type='DistributedGroupSampler'),
            nonshuffler_sampler=cfg.data.nonshuffler_sampler,  # dict(type='DistributedSampler'),
            runner_type=cfg.runner
        ) for ds in dataset
    ]
    
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        model = DistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
        if eval_model is not None:
            eval_model = DistributedDataParallel(
                eval_model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
    else:
        model = DataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)
        if eval_model is not None:
            eval_model = DataParallel(
                eval_model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    if 'runner' not in cfg:
        cfg.runner = {
            'type': 'EpochBasedRunner',
            'max_epochs': cfg.total_epochs
        }
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)
    else:
        if 'total_epochs' in cfg:
            assert cfg.total_epochs == cfg.runner.max_epochs
    if eval_model is not None:
        runner = build_runner(
            cfg.runner,
            default_args=dict(
                model=model,
                eval_model=eval_model,
                optimizer=optimizer,
                work_dir=cfg.work_dir,
                logger=logger,
                meta=meta))
    else:
        runner = build_runner(
            cfg.runner,
            default_args=dict(
                model=model,
                optimizer=optimizer,
                work_dir=cfg.work_dir,
                logger=logger,
                meta=meta))

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))
    
    
    if distributed:
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        # Support batch_size > 1 in validation
        val_samples_per_gpu = cfg.data.val.pop('samples_per_gpu', 1)
        if val_samples_per_gpu > 1:
            assert False
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.val.pipeline = replace_ImageToTensor(
                cfg.data.val.pipeline)
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))

        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=val_samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False,
            shuffler_sampler=cfg.data.shuffler_sampler,  # dict(type='DistributedGroupSampler'),
            nonshuffler_sampler=cfg.data.nonshuffler_sampler,  # dict(type='DistributedSampler'),
        )
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = (cfg.runner['type'] != 'IterBasedRunner' or  cfg.runner['type'] != 'RLIterBasedRunner')
        eval_cfg['jsonfile_prefix'] = osp.join('val', cfg.work_dir, time.ctime().replace(' ','_').replace(':','_'))
        eval_hook = CustomDistEvalHook if distributed else EvalHook
        runner.register_hook(eval_hook(val_dataloader, test_fn=custom_multi_gpu_test, **eval_cfg))
    

    # user-defined hooks
    if cfg.get('custom_hooks', None):
        custom_hooks = cfg.custom_hooks
        assert isinstance(custom_hooks, list), \
            f'custom_hooks expect list type, but got {type(custom_hooks)}'
        for hook_cfg in cfg.custom_hooks:
            assert isinstance(hook_cfg, dict), \
                'Each item in custom_hooks expects dict type, but got ' \
                f'{type(hook_cfg)}'
            hook_cfg = hook_cfg.copy()
            priority = hook_cfg.pop('priority', 'NORMAL')
            hook = build_from_cfg(hook_cfg, HOOKS)
            runner.register_hook(hook, priority=priority)

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif args.load_from:
        runner.load_checkpoint(args.load_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)

    if cfg.runner['type'] == 'RLIterBasedRunner':
        inference_only_pipelines = []
        for inference_only_pipeline in cfg.inference_only_pipeline:
                inference_only_pipelines.append(inference_only_pipeline)
        inference_only_pipelines = Compose(inference_only_pipelines)
        # # @hyfu：ablation study 在这里配置
        # if hasattr(cfg, 'kl_coef'):
        #     args.kl_coef = cfg.kl_coef
        # if hasattr(cfg, 'no_use_kl'):
        #     args.no_use_kl = cfg.no_use_kl
        # if hasattr(cfg, 'use_entroy'):
        #     args.use_entroy = cfg.use_entroy
        runner.run(data_loaders, cfg.workflow, inference_only_pipelines, args)
    else:
        runner.run(data_loaders, cfg.workflow)

def custom_collect_dataset(model,
                   cfg,
                   args=None,
                   ):
    data_loaders = None 
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    runner = build_runner(
            cfg.runner,
            default_args=dict(model=model,logger=logger))
       
    if cfg.load_from and args.checkpoint is None:
        runner.load_checkpoint(cfg.load_from)
    elif args.weight is not None:
        runner.load_checkpoint(args.weight)

    if cfg.runner['type'] == 'RLIterBasedRunner':
        # 处理数据的
        if hasattr(cfg, 'PENALTY_CONFIG'): # 消融
            args.PENALTY_CONFIG = cfg.PENALTY_CONFIG
        inference_only_pipelines = []
        for inference_only_pipeline in cfg.inference_only_pipeline:
                inference_only_pipelines.append(inference_only_pipeline)
        inference_only_pipelines = Compose(inference_only_pipelines)
        runner.run(data_loaders, cfg.workflow, inference_only_pipelines, args)
    else:
        runner.run(data_loaders, cfg.workflow)