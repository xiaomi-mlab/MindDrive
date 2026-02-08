# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import platform
import shutil
import time
import warnings
from typing import Callable, Dict, List, Optional, Tuple, Union, no_type_check
import socket
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader

import mmcv
from .base_runner import BaseRunner
from .builder import RUNNERS
from .hooks import IterTimerHook
from ..utils import save_checkpoint, is_list_of, symlink, get_host_info

from mmcv.runner.buffers import RolloutBuffer
import torch.distributed as dist
import numpy as np
import cv2
import os
from tabulate import tabulate

from team_code.carla_env.carla_env_scenario import CarlaScenarioEnv, TickRuntimeError

from mmcv.parallel.collate import collate as mm_collate_to_batch_form
import copy
from rl_projects.leaderboard.utils.result_writer import ResultOutputProvider
import pickle
import torch.nn.functional as F
from rl_projects.utils.carla_utils import kill_existing_carla, launch_carla
hostname = socket.gethostname()
def tensor_to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: tensor_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [tensor_to_device(v, device) for v in data]
    elif isinstance(data, tuple):
        return tuple(tensor_to_device(v, device) for v in data)
    else:
        return data 

class IterLoader:

    def __init__(self, dataloader: DataLoader):
        self._dataloader = dataloader
        self.iter_loader = iter(self._dataloader)
        self._epoch = 0

    @property
    def epoch(self) -> int:
        return self._epoch

    def __next__(self):
        try:
            data = next(self.iter_loader)
        except StopIteration:
            self._epoch += 1
            if hasattr(self._dataloader.sampler, 'set_epoch'):
                self._dataloader.sampler.set_epoch(self._epoch)
            time.sleep(2)  # Prevent possible deadlock during epoch transition
            self.iter_loader = iter(self._dataloader)
            data = next(self.iter_loader)

        return data

    def __len__(self):
        return len(self._dataloader)


@RUNNERS.register_module()
class IterBasedRunner(BaseRunner):
    """Iteration-based Runner.

    This runner train models iteration by iteration.
    """

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._epoch = data_loader.epoch
        data_batch = next(data_loader)
        self.data_batch = data_batch
        self.call_hook('before_train_iter')
        outputs = self.model(data_batch, return_loss=True, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError('model.train_step() must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs
        self.call_hook('after_train_iter')
        del self.data_batch
        self._inner_iter += 1
        self._iter += 1

    @torch.no_grad()
    def val(self, data_loader, **kwargs):
        self.model.eval()
        self.mode = 'val'
        self.data_loader = data_loader
        data_batch = next(data_loader)
        self.data_batch = data_batch
        self.call_hook('before_val_iter')
        outputs = self.model.val_step(data_batch, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError('model.val_step() must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs
        self.call_hook('after_val_iter')
        del self.data_batch
        self._inner_iter += 1

    def run(self,
            data_loaders: List[DataLoader],
            workflow: List[Tuple[str, int]],
            max_iters: Optional[int] = None,
            **kwargs) -> None:
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, iters) to specify the
                running order and iterations. E.g, [('train', 10000),
                ('val', 1000)] means running 10000 iterations for training and
                1000 iterations for validation, iteratively.
        """
        assert isinstance(data_loaders, list)
        assert mmcv.is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow)
        if max_iters is not None:
            warnings.warn(
                'setting max_iters in run is deprecated, '
                'please set max_iters in runner_config', DeprecationWarning)
            self._max_iters = max_iters
        assert self._max_iters is not None, (
            'max_iters must be specified during instantiation')

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('Hooks will be executed in the following order:\n%s',
                         self.get_hook_info())
        self.logger.info('workflow: %s, max: %d iters', workflow,
                         self._max_iters)
        self.call_hook('before_run')

        iter_loaders = [IterLoader(x) for x in data_loaders]

        self.call_hook('before_epoch')

        while self.iter < self._max_iters:
            for i, flow in enumerate(workflow):
                self._inner_iter = 0
                mode, iters = flow
                if not isinstance(mode, str) or not hasattr(self, mode):
                    raise ValueError(
                        'runner has no method named "{}" to run a workflow'.
                        format(mode))
                iter_runner = getattr(self, mode)
                for _ in range(iters):
                    if mode == 'train' and self.iter >= self._max_iters:
                        break
                    iter_runner(iter_loaders[i], **kwargs)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_epoch')
        self.call_hook('after_run')

    @no_type_check
    def resume(self,
               checkpoint: str,
               resume_optimizer: bool = True,
               map_location: Union[str, Callable] = 'default') -> None:
        """Resume model from checkpoint.

        Args:
            checkpoint (str): Checkpoint to resume from.
            resume_optimizer (bool, optional): Whether resume the optimizer(s)
                if the checkpoint file includes optimizer(s). Default to True.
            map_location (str, optional): Same as :func:`torch.load`.
                Default to 'default'.
        """
        if map_location == 'default':
            device_id = torch.cuda.current_device()
            checkpoint = self.load_checkpoint(
                checkpoint,
                map_location=lambda storage, loc: storage.cuda(device_id))
        else:
            checkpoint = self.load_checkpoint(
                checkpoint, map_location=map_location)

        self._epoch = checkpoint['meta']['epoch']
        self._iter = checkpoint['meta']['iter']
        self._inner_iter = checkpoint['meta']['iter']
        if 'optimizer' in checkpoint and resume_optimizer:
            if isinstance(self.optimizer, Optimizer):
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            elif isinstance(self.optimizer, dict):
                for k in self.optimizer.keys():
                    self.optimizer[k].load_state_dict(
                        checkpoint['optimizer'][k])
            else:
                raise TypeError(
                    'Optimizer should be dict or torch.optim.Optimizer '
                    f'but got {type(self.optimizer)}')

        self.logger.info(f'resumed from epoch: {self.epoch}, iter {self.iter}')

    def save_checkpoint(  # type: ignore
            self,
            out_dir: str,
            filename_tmpl: str = 'iter_{}.pth',
            meta: Optional[Dict] = None,
            save_optimizer: bool = True,
            create_symlink: bool = True) -> None:
        """Save checkpoint to file.

        Args:
            out_dir (str): Directory to save checkpoint files.
            filename_tmpl (str, optional): Checkpoint file template.
                Defaults to 'iter_{}.pth'.
            meta (dict, optional): Metadata to be saved in checkpoint.
                Defaults to None.
            save_optimizer (bool, optional): Whether save optimizer.
                Defaults to True.
            create_symlink (bool, optional): Whether create symlink to the
                latest checkpoint file. Defaults to True.
        """
        if meta is None:
            meta = {}
        elif not isinstance(meta, dict):
            raise TypeError(
                f'meta should be a dict or None, but got {type(meta)}')
        if self.meta is not None:
            meta.update(self.meta)
            # Note: meta.update(self.meta) should be done before
            # meta.update(epoch=self.epoch + 1, iter=self.iter) otherwise
            # there will be problems with resumed checkpoints.
            # More details in https://github.com/open-mmlab/mmcv/pull/1108
        meta.update(epoch=self.epoch + 1, iter=self.iter)

        filename = filename_tmpl.format(self.iter + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        # in some environments, `os.symlink` is not supported, you may need to
        # set `create_symlink` to False
        if create_symlink:
            dst_file = osp.join(out_dir, 'latest.pth')
            if platform.system() != 'Windows':
                mmcv.symlink(filename, dst_file)
            else:
                shutil.copy(filepath, dst_file)

    def register_training_hooks(self,
                                lr_config,
                                optimizer_config=None,
                                checkpoint_config=None,
                                log_config=None,
                                momentum_config=None,
                                custom_hooks_config=None):
        """Register default hooks for iter-based training.

        Checkpoint hook, optimizer stepper hook and logger hooks will be set to
        `by_epoch=False` by default.

        Default hooks include:

        +----------------------+-------------------------+
        | Hooks                | Priority                |
        +======================+=========================+
        | LrUpdaterHook        | VERY_HIGH (10)          |
        +----------------------+-------------------------+
        | MomentumUpdaterHook  | HIGH (30)               |
        +----------------------+-------------------------+
        | OptimizerStepperHook | ABOVE_NORMAL (40)       |
        +----------------------+-------------------------+
        | CheckpointSaverHook  | NORMAL (50)             |
        +----------------------+-------------------------+
        | IterTimerHook        | LOW (70)                |
        +----------------------+-------------------------+
        | LoggerHook(s)        | VERY_LOW (90)           |
        +----------------------+-------------------------+
        | CustomHook(s)        | defaults to NORMAL (50) |
        +----------------------+-------------------------+

        If custom hooks have same priority with default hooks, custom hooks
        will be triggered after default hooks.
        """
        if checkpoint_config is not None:
            checkpoint_config.setdefault('by_epoch', False)  # type: ignore
        if lr_config is not None:
            lr_config.setdefault('by_epoch', False)  # type: ignore
        if log_config is not None:
            for info in log_config['hooks']:
                info.setdefault('by_epoch', False)
        super().register_training_hooks(
            lr_config=lr_config,
            momentum_config=momentum_config,
            optimizer_config=optimizer_config,
            checkpoint_config=checkpoint_config,
            log_config=log_config,
            timer_config=IterTimerHook(),
            custom_hooks_config=custom_hooks_config)

@RUNNERS.register_module()
class RLIterBasedRunner(BaseRunner):
    def __init__(self, il_step=40, n_steps=256, cache_obs=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_obs = None
        self._last_obs_path = None
        self._last_episode_starts = None
        self.il_step = il_step
        self.n_steps = n_steps # TODO@Jianfeng
        self.num_timesteps = 0
        self.gamma = 0.99
        self.frame_rate = 10
        self.rollout_buffer = RolloutBuffer(buffer_size=self.n_steps, cache_obs=cache_obs)
        self.normalize_advantage = True
        self.clip_range = 0.2
        self.clip_range_vf = None
        self.ent_coef = 0.0
        self.vf_coef = 0.5
        self.kl_coef = 0.5
        self.ppo_batch_size = 16  # TODO@Jianfeng
        self.rollout_generator = None
        self.scene_counts = {}
        self.scene_metrics = {}
        self.cache_obs = cache_obs
    

    def collect_rollouts_scenario(self, env, rollout_buffer, n_rollout_steps: int, il_rollout=None, clip_index=None, evaluate=False, PENALTY_CONFIG=None) -> bool:
        device = next(self.model.parameters()).device
        all_scence_done = False
        n_steps = 0
        episode_steps = 0
        max_episode_steps = env.max_tick_count
        rollout_buffer.reset()
        env.reset(self._inner_iter)
        self.rollout_generator = None
        
        self._last_episode_starts = True

        while True:
            if self._last_episode_starts:
                self._check = env.soft_reset() 
                if self._check is None:
                    all_scence_done = True
                    print('All Scence Done!!!!')
                    break
                self._last_obs, rewards, done, infos = env.step(None, None, None)
                rollout_buffer.reset()
                episode_steps = 0
 
            with torch.no_grad():
                self._last_obs = self.inference_only_pipeline(self._last_obs)
                input_data_batch = mm_collate_to_batch_form([self._last_obs], samples_per_gpu=1)
                input_data_batch = tensor_to_device(input_data_batch, device)                   
                outputs = self.model(input_data_batch, return_loss=False) 
            
            out_truck = outputs[0]['pts_bbox']['ego_fut_preds'].cpu().numpy()
            out_truck_path = outputs[0]['pts_bbox']['pw_ego_fut_pred'].cpu().numpy()

            command_speed = None 
            command_path = outputs[0]['pts_bbox']['path_value']    
            # action
            actions = outputs[0]['pts_bbox']['speed_value']
            # LLM input_id
            meta_action_info = outputs[0]['pts_bbox']['meta_action_info']
            ppo_info = outputs[0]['pts_bbox']['ppo_info']
            
            ref_action_log_probs = ppo_info['reference_action_log_prob']
            values = ppo_info['values']
            
            new_obs, rewards, done, infos = env.step(out_truck, out_truck_path, self._last_obs, command_speed, command_path, outputs, PENALTY_CONFIG)

            n_steps += 1
            episode_steps += 1
            
            if episode_steps >= max_episode_steps+1:
                print("Warning: Episode timed out!")
                done = True
            
            rollout_buffer.add(actions, rewards, self._last_episode_starts, values, ref_action_log_probs, meta_action_info)
            
            # value net
            self._last_obs = new_obs
            self._last_episode_starts = done 

            if done:
                env._register_statistics(env.route_index, env.entry_status, env.crash_message)
                env.statistics_manager.save_progress(env.route_indexer.index, env.route_indexer.total)
                env.statistics_manager.write_statistics()
                rollout_buffer.compute_returns_and_advantage(last_values=None, dones=done) 
                episode_save_path = f"./{env.save_path}/rollout_buffer_{env.config.name}_{env.config.repetition_index}.pkl"
                rollout_buffer.save_current_episode(rollout_buffer, episode_save_path)


        return True
    

    @torch.no_grad()
    def rollout(self, args=None, **kwargs):
        self.model.eval()
        self.mode = 'rollout'
        self.call_hook('before_val_iter')
        env = CarlaScenarioEnv(routes =args.routes,repetitions=args.repetitions,port=args.port, traffic_manager_port=args.traffic_manager_port,checkpoint=args.checkpoint, resume = args.resume)
        PENALTY_CONFIG= None
        if hasattr(args,'PENALTY_CONFIG'):
            PENALTY_CONFIG = args.PENALTY_CONFIG

        self.collect_rollouts_scenario(env, self.rollout_buffer, n_rollout_steps=self.n_steps, evaluate=False,PENALTY_CONFIG=PENALTY_CONFIG)       
        self._iter += 1

    def merge_obs_data(self, obs_list):
        if self.cache_obs:
            output_dict = obs_list[0]
            if len(obs_list) > 1:
                for i in range(1,len(obs_list)):
                    for key in output_dict.keys():
                        output_dict[key].extend(obs_list[i][key])
        else:
            output_dict = {k: [x[k][0] for x in obs_list] for k in obs_list[0].keys()}
        return output_dict

    def ppo_train(self, data_loader, args, **kwargs):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mode = 'ppo_train'
        self.data_loader = data_loader
        losses = dict()
        continue_training = True
        clip_range = self.clip_range
        self.model.train()
        batch = next(data_loader)
        batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
        actions = batch['actions']               
        values = batch['values']                 
        ref_log_probs = batch['ref_log_probs']    
        advantages = batch['advantages'].squeeze(-1)      
        returns = batch['returns']     

        inputs_embeds = batch['inputs_embeds']
        new_input_ids = batch['new_input_ids']        

        meta_action_info ={
            'inputs_embeds':inputs_embeds,
            'new_input_ids':new_input_ids,
        }


        action_log_probs, action_lang_log_probs, state_values= self.model(meta_action_info, return_loss=False, is_rl_training=True)

        if self.normalize_advantage and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        selected_action_log_probs = action_log_probs.gather(1, actions.long()).squeeze(1)
        selected_ref_log_probs = ref_log_probs.gather(1, actions.long()).squeeze(1)
        ratio = torch.exp(selected_action_log_probs - selected_ref_log_probs)

        # clipped surrogate loss
        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
        policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

        losses['ppo_loss'] = policy_loss

        if self.clip_range_vf is None:
            pred_values = state_values 
        else:
            # pass
            pred_values = values + torch.clamp(
                pred_values - values, -self.clip_range_vf, self.clip_range_vf
            )
        value_loss = F.mse_loss(pred_values, returns)
        losses['value_loss'] = self.vf_coef * value_loss

        if hasattr(args, 'no_use_kl'):
            no_use_kl = args.no_use_kl
        else:
            no_use_kl = False

        if hasattr(args, 'use_entroy'):
            use_entroy = args.use_entroy
        else:
            use_entroy = False  

        if not no_use_kl:
            kl_loss = F.kl_div(
                action_log_probs,
                ref_log_probs, 
                log_target=True, 
                reduction='batchmean'
            )
            if hasattr(args, 'kl_coef'):
                self.kl_coef = args.kl_coef
            losses['kl_loss'] = self.kl_coef * kl_loss
        if use_entroy:
            probs = torch.exp(action_log_probs)  # 概率
            policy_entropy = torch.distributions.Categorical(probs=probs).entropy().mean()
            entropy_coef = 0.01
            losses['entropy_loss'] = -entropy_coef * policy_entropy

        total_loss = sum(_value for _key, _value in losses.items()
                        if 'loss' in _key)

        losses['loss'] = total_loss

        for loss_name, loss_value in losses.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            losses[loss_name] = loss_value.item()
        
        self.log_buffer.update(losses, self.ppo_batch_size)

        self.outputs = dict(loss = total_loss, log_vars = losses, num_samples = self.ppo_batch_size)
        self.call_hook('after_train_iter')
        self._inner_iter += 1
        self._iter += 1

    def rollout_train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'rollout_train'
        losses = dict()
        continue_training = True
        clip_range = self.clip_range
        
        actions, values, ref_log_probs, advantages, returns, _ = next(self.rollout_buffer_il.get(self.il_batch_size)) 

        batch_observations = self.merge_obs_data(observations)
        batch_observations['timestamp'] = [ts.squeeze(0) for ts in batch_observations['timestamp']]
        output = self.model(batch_observations, return_loss=True, is_rl_training=False, simulator_il=True)

        losses['il_planning_loss'] = output['loss'].item()
        self.outputs = output
        if 'log_vars' in output:
            self.log_buffer.update(output['log_vars'], output['num_samples'])
        self.call_hook('after_train_iter')
        # del self.data_batch
        self._inner_iter += 1
        self._iter += 1



    def train(self, data_loader, cfg=None, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._epoch = data_loader.epoch
        data_batch = next(data_loader)
        self.data_batch = data_batch
        self.call_hook('before_train_iter')
        # outputs = self.model(data_batch, self.optimizer, **kwargs)
        outputs = self.model(data_batch, return_loss=True, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError('model.train_step() must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs
        self.call_hook('after_train_iter')
        del self.data_batch
        self._inner_iter += 1
        self._iter += 1

    @torch.no_grad()
    def val(self, data_loader, **kwargs):
        self.model.eval()
        self.mode = 'val'
        self.data_loader = data_loader
        data_batch = next(data_loader)
        self.data_batch = data_batch
        self.call_hook('before_val_iter')
        outputs = self.model.val_step(data_batch, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError('model.val_step() must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs
        self.call_hook('after_val_iter')
        del self.data_batch
        self._inner_iter += 1

    def run_il(self,
            data_loaders: List[DataLoader],
            workflow: List[Tuple[str, int]],
            max_iters: Optional[int] = None,
            **kwargs) -> None:
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, iters) to specify the
                running order and iterations. E.g, [('train', 10000),
                ('val', 1000)] means running 10000 iterations for training and
                1000 iterations for validation, iteratively.
        """
        assert isinstance(data_loaders, list)
        assert mmcv.is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow)
        if max_iters is not None:
            warnings.warn(
                'setting max_iters in run is deprecated, '
                'please set max_iters in runner_config', DeprecationWarning)
            self._max_iters = max_iters
        assert self._max_iters is not None, (
            'max_iters must be specified during instantiation')

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('Hooks will be executed in the following order:\n%s',
                         self.get_hook_info())
        self.logger.info('workflow: %s, max: %d iters', workflow,
                         self._max_iters)
        self.call_hook('before_run')

        iter_loaders = [IterLoader(x) for x in data_loaders]

        self.call_hook('before_epoch')

        while self.iter < self._max_iters:
            for i, flow in enumerate(workflow):
                self._inner_iter = 0
                mode, iters = flow
                if not isinstance(mode, str) or not hasattr(self, mode):
                    raise ValueError(
                        'runner has no method named "{}" to run a workflow'.
                        format(mode))
                iter_runner = getattr(self, mode)
                for _ in range(iters):
                    if mode == 'train' and self.iter >= self._max_iters:
                        break
                    iter_runner(iter_loaders[i], **kwargs)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_epoch')
        self.call_hook('after_run')

    def run(self,
            data_loaders: List[DataLoader],
            workflow: List[Tuple[str, int]],
            inference_only_pipeline,
            args,
            max_iters: Optional[int] = None,
            **kwargs) -> None:
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, iters) to specify the
                running order and iterations. E.g, [('train', 10000),
                ('val', 1000)] means running 10000 iterations for training and
                1000 iterations for validation, iteratively.
        """
        if not args.use_carla:
            assert isinstance(data_loaders, list)
            assert mmcv.is_list_of(workflow, tuple)
            assert len(data_loaders) == len(workflow)
            if max_iters is not None:
                warnings.warn(
                    'setting max_iters in run is deprecated, '
                    'please set max_iters in runner_config', DeprecationWarning)
                self._max_iters = max_iters
            assert self._max_iters is not None, (
                'max_iters must be specified during instantiation')

            work_dir = self.work_dir if self.work_dir is not None else 'NONE'
            self.logger.info('Start running, host: %s, work_dir: %s',
                            get_host_info(), work_dir)
            self.logger.info('Hooks will be executed in the following order:\n%s',
                            self.get_hook_info())
            self.logger.info('workflow: %s, max: %d iters', workflow,
                            self._max_iters)
            self.call_hook('before_run')

            from mmcv.datasets import B2D_minddrive_Dataset, RL_minddrive_Dataset
            iter_loaders = [IterLoader(x) if isinstance(x.dataset, RL_minddrive_Dataset) else x for x in data_loaders]

            self.call_hook('before_epoch')
        else:
            self.inference_only_pipeline = inference_only_pipeline
    
        while self.iter < self._max_iters:
            for i, flow in enumerate(workflow):
                self._inner_iter = 0
                mode, iters = flow # 'train', 1
                if not isinstance(mode, str) or not hasattr(self, mode):
                    raise ValueError(
                        'runner has no method named "{}" to run a workflow'.
                        format(mode))
                iter_runner = getattr(self, mode)
                for _ in range(iters):
                    if mode == 'train' and self.iter >= self._max_iters:
                        break
                    if mode == 'rollout':
                        iter_runner(args, **kwargs)
                    else:
                        iter_runner(iter_loaders[i], args, **kwargs)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_epoch')
        self.call_hook('after_run')

    @no_type_check
    def resume(self,
               checkpoint: str,
               resume_optimizer: bool = True,
               map_location: Union[str, Callable] = 'default') -> None:
        """Resume model from checkpoint.

        Args:
            checkpoint (str): Checkpoint to resume from.
            resume_optimizer (bool, optional): Whether resume the optimizer(s)
                if the checkpoint file includes optimizer(s). Default to True.
            map_location (str, optional): Same as :func:`torch.load`.
                Default to 'default'.
        """
        if map_location == 'default':
            device_id = torch.cuda.current_device()
            checkpoint = self.load_checkpoint(
                checkpoint,
                map_location=lambda storage, loc: storage.cuda(device_id))
        else:
            checkpoint = self.load_checkpoint(
                checkpoint, map_location=map_location)

        self._epoch = checkpoint['meta']['epoch']
        self._iter = checkpoint['meta']['iter']
        self._inner_iter = checkpoint['meta']['iter']
        if 'optimizer' in checkpoint and resume_optimizer:
            if isinstance(self.optimizer, Optimizer):
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            elif isinstance(self.optimizer, dict):
                for k in self.optimizer.keys():
                    self.optimizer[k].load_state_dict(
                        checkpoint['optimizer'][k])
            else:
                raise TypeError(
                    'Optimizer should be dict or torch.optim.Optimizer '
                    f'but got {type(self.optimizer)}')

        self.logger.info(f'resumed from epoch: {self.epoch}, iter {self.iter}')

    def save_checkpoint(  # type: ignore
            self,
            out_dir: str,
            filename_tmpl: str = 'iter_{}.pth',
            meta: Optional[Dict] = None,
            save_optimizer: bool = True,
            create_symlink: bool = True) -> None:
        """Save checkpoint to file.

        Args:
            out_dir (str): Directory to save checkpoint files.
            filename_tmpl (str, optional): Checkpoint file template.
                Defaults to 'iter_{}.pth'.
            meta (dict, optional): Metadata to be saved in checkpoint.
                Defaults to None.
            save_optimizer (bool, optional): Whether save optimizer.
                Defaults to True.
            create_symlink (bool, optional): Whether create symlink to the
                latest checkpoint file. Defaults to True.
        """
        if meta is None:
            meta = {}
        elif not isinstance(meta, dict):
            raise TypeError(
                f'meta should be a dict or None, but got {type(meta)}')
        if self.meta is not None:
            meta.update(self.meta)
            # Note: meta.update(self.meta) should be done before
            # meta.update(epoch=self.epoch + 1, iter=self.iter) otherwise
            # there will be problems with resumed checkpoints.
            # More details in https://github.com/open-mmlab/mmcv/pull/1108
        meta.update(epoch=self.epoch + 1, iter=self.iter)

        filename = filename_tmpl.format(self.iter + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        # in some environments, `os.symlink` is not supported, you may need to
        # set `create_symlink` to False
        if create_symlink:
            dst_file = osp.join(out_dir, 'latest.pth')
            if platform.system() != 'Windows':
                mmcv.symlink(filename, dst_file)
            else:
                shutil.copy(filepath, dst_file)


    def register_training_hooks(self,
                                lr_config,
                                optimizer_config=None,
                                checkpoint_config=None,
                                log_config=None,
                                momentum_config=None,
                                custom_hooks_config=None):
        """Register default hooks for iter-based training.

        Checkpoint hook, optimizer stepper hook and logger hooks will be set to
        `by_epoch=False` by default.

        Default hooks include:

        +----------------------+-------------------------+
        | Hooks                | Priority                |
        +======================+=========================+
        | LrUpdaterHook        | VERY_HIGH (10)          |
        +----------------------+-------------------------+
        | MomentumUpdaterHook  | HIGH (30)               |
        +----------------------+-------------------------+
        | OptimizerStepperHook | ABOVE_NORMAL (40)       |
        +----------------------+-------------------------+
        | CheckpointSaverHook  | NORMAL (50)             |
        +----------------------+-------------------------+
        | IterTimerHook        | LOW (70)                |
        +----------------------+-------------------------+
        | LoggerHook(s)        | VERY_LOW (90)           |
        +----------------------+-------------------------+
        | CustomHook(s)        | defaults to NORMAL (50) |
        +----------------------+-------------------------+

        If custom hooks have same priority with default hooks, custom hooks
        will be triggered after default hooks.
        """
        if checkpoint_config is not None:
            checkpoint_config.setdefault('by_epoch', False)  # type: ignore
        if lr_config is not None:
            lr_config.setdefault('by_epoch', False)  # type: ignore
        if log_config is not None:
            for info in log_config['hooks']:
                info.setdefault('by_epoch', False)
        super().register_training_hooks(
            lr_config=lr_config,
            momentum_config=momentum_config,
            optimizer_config=optimizer_config,
            checkpoint_config=checkpoint_config,
            log_config=log_config,
            timer_config=IterTimerHook(),
            custom_hooks_config=custom_hooks_config)