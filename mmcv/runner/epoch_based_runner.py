# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import platform
import shutil
import time
import warnings

import torch

from .base_runner import BaseRunner
from .builder import RUNNERS
from ..utils import save_checkpoint, is_list_of, symlink, get_host_info
import torch.nn.functional as F
import torch.distributed as dist




@RUNNERS.register_module()
class EpochBasedRunner(BaseRunner):
    """Epoch-based Runner.

    This runner train models epoch by epoch.
    """

    def run_iter(self, data_batch, train_mode, **kwargs):
        if self.batch_processor is not None:
            outputs = self.batch_processor(
                self.model, data_batch, train_mode=train_mode, **kwargs)
        elif train_mode:
            outputs = self.model(data_batch, return_loss=train_mode, **kwargs)
        else:
            outputs = self.model.val_step(data_batch, self.optimizer, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError('"batch_processor()" or "model.train_step()"'
                            'and "model.val_step()" must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(self.data_loader)
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            self.call_hook('before_train_iter')
            self.run_iter(data_batch, train_mode=True, **kwargs)
            self.call_hook('after_train_iter')
            self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1

    @torch.no_grad()
    def val(self, data_loader, **kwargs):
        self.model.eval()
        self.mode = 'val'
        self.data_loader = data_loader
        self.call_hook('before_val_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            self.call_hook('before_val_iter')
            self.run_iter(data_batch, train_mode=False)
            self.call_hook('after_val_iter')
            

        self.call_hook('after_val_epoch')

    def ppo_train(self, data_loader, **kwargs):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mode = 'ppo_train'
        self.model.train()  
        self.data_loader = data_loader
        self.call_hook('before_train_epoch')
        self._inner_iter = 0
        self.ppo_batch_size = data_loader.batch_size
        for batch in data_loader:  
            batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
            actions = batch['actions']               
            values = batch['values']                 
            ref_log_probs = batch['ref_log_probs']    
            advantages = batch['advantages'].squeeze(-1)      
            returns = batch['returns']

            meta_action_info = {
                'inputs_embeds': batch['inputs_embeds'],
                'new_input_ids': batch['new_input_ids'],
            }

            action_log_probs, _, state_values = self.model(meta_action_info, return_loss=False, is_rl_training=True)

            # Advantage
            if len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            selected_action_log_probs = action_log_probs.gather(1, actions.long()).squeeze(1)
            selected_ref_log_probs = ref_log_probs.gather(1, actions.long()).squeeze(1)
            ratio = torch.exp(selected_action_log_probs - selected_ref_log_probs)

            clip_range = 0.2
            policy_loss_1 = advantages * ratio
            policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
            policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
            losses = {'ppo_loss': policy_loss}

            # Value Loss
            pred_values = state_values 
            
            value_loss = F.mse_loss(pred_values, returns)
            self.vf_coef= 0.5
            losses['value_loss'] = self.vf_coef * value_loss

            self.kl_coef=0.5
            kl_loss = F.kl_div(
                    action_log_probs, ref_log_probs, log_target=True, reduction='batchmean'
                )
            losses['kl_loss'] = self.kl_coef * kl_loss

            total_loss = sum(_value for _key, _value in losses.items() if 'loss' in _key)
            losses['loss'] = total_loss

            for loss_name, loss_value in losses.items():
                if dist.is_available() and dist.is_initialized():
                    loss_value = loss_value.data.clone()
                    dist.all_reduce(loss_value.div_(dist.get_world_size()))
                losses[loss_name] = loss_value.item()
    
            self.log_buffer.update(losses, self.ppo_batch_size)

            self.outputs = dict(loss=total_loss, log_vars=losses, num_samples=self.ppo_batch_size)
            self.call_hook('after_train_iter')  
            self._inner_iter += 1

        self.call_hook('after_train_epoch') 
        self._epoch += 1 


    def run(self, data_loaders, workflow, max_epochs=None, **kwargs):
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs. E.g, [('train', 2), ('val', 1)] means
                running 2 epochs for training and 1 epoch for validation,
                iteratively.
        """
        assert isinstance(data_loaders, list)
        assert is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow)
        if max_epochs is not None:
            warnings.warn(
                'setting max_epochs in run is deprecated, '
                'please set max_epochs in runner_config', DeprecationWarning)
            self._max_epochs = max_epochs

        assert self._max_epochs is not None, (
            'max_epochs must be specified during instantiation')

        for i, flow in enumerate(workflow):
            mode, epochs = flow
            if mode == 'train':
                self._max_iters = self._max_epochs * len(data_loaders[i])
                break

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('Hooks will be executed in the following order:\n%s',
                         self.get_hook_info())
        self.logger.info('workflow: %s, max: %d epochs', workflow,
                         self._max_epochs)
        self.call_hook('before_run')

        while self.epoch < self._max_epochs:
            for i, flow in enumerate(workflow):
                mode, epochs = flow
                if isinstance(mode, str):  # self.train()
                    if not hasattr(self, mode):
                        raise ValueError(
                            f'runner has no method named "{mode}" to run an '
                            'epoch')
                    epoch_runner = getattr(self, mode)
                else:
                    raise TypeError(
                        'mode in workflow must be a str, but got {}'.format(
                            type(mode)))

                for _ in range(epochs):
                    if mode == 'train' and self.epoch >= self._max_epochs:
                        break
                    epoch_runner(data_loaders[i], **kwargs)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')

    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl='epoch_{}.pth',
                        save_optimizer=True,
                        meta=None,
                        create_symlink=True):
        """Save the checkpoint.

        Args:
            out_dir (str): The directory that checkpoints are saved.
            filename_tmpl (str, optional): The checkpoint filename template,
                which contains a placeholder for the epoch number.
                Defaults to 'epoch_{}.pth'.
            save_optimizer (bool, optional): Whether to save the optimizer to
                the checkpoint. Defaults to True.
            meta (dict, optional): The meta information to be saved in the
                checkpoint. Defaults to None.
            create_symlink (bool, optional): Whether to create a symlink
                "latest.pth" to point to the latest checkpoint.
                Defaults to True.
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

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        # in some environments, `os.symlink` is not supported, you may need to
        # set `create_symlink` to False
        if create_symlink:
            dst_file = osp.join(out_dir, 'latest.pth')
            if platform.system() != 'Windows':
                symlink(filename, dst_file)
            else:
                shutil.copy(filepath, dst_file)



@RUNNERS.register_module()
class EpochBasedRunner_video(EpochBasedRunner):
    
    ''' 
    # basic logic
    
    input_sequence = [a, b, c] # given a sequence of samples
    
    prev_bev = None
    for each in input_sequcene[:-1]
        prev_bev = eval_model(each, prev_bev)) # inference only.
    
    model(input_sequcene[-1], prev_bev) # train the last sample.
    '''
    
    def __init__(self,
                 model,
                 eval_model=None,
                 batch_processor=None,
                 optimizer=None,
                 work_dir=None,
                 logger=None,
                 meta=None,
                 keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'],
                 max_iters=None,
                 max_epochs=None):
        super().__init__(model,
                 batch_processor,
                 optimizer,
                 work_dir,
                 logger,
                 meta,
                 max_iters,
                 max_epochs)
        keys.append('img_metas')
        self.keys = keys
        self.eval_model = eval_model
        self.eval_model.eval()
    
    def run_iter(self, data_batch, train_mode, **kwargs):
        if self.batch_processor is not None:
            assert False
            # outputs = self.batch_processor(
            #     self.model, data_batch, train_mode=train_mode, **kwargs)
        elif train_mode:

            num_samples = data_batch['img'].data[0].size(1)
            data_list = []
            prev_bev = None
            for i in range(num_samples):
                data = {}
                for key in self.keys:
                    if key not in ['img_metas', 'img', 'points']:
                        data[key] = data_batch[key]
                    else:
                        if key == 'img':
                            data['img'] = DataContainer(data=[data_batch['img'].data[0][:, i]], cpu_only=data_batch['img'].cpu_only, stack=True)
                        elif key == 'img_metas':
                            data['img_metas'] = DataContainer(data=[[each[i] for each in data_batch['img_metas'].data[0]]], cpu_only=data_batch['img_metas'].cpu_only)
                        else:
                            assert False
                data_list.append(data)
            with torch.no_grad():
                for i in range(num_samples-1):
                    if i>0: data_list[i]['prev_bev'] = DataContainer(data=[prev_bev], cpu_only=False)
                    prev_bev = self.eval_model.val_step(data_list[i], self.optimizer, **kwargs)
            
            data_list[-1]['prev_bev'] = DataContainer(data=[prev_bev], cpu_only=False)
            outputs = self.model.train_step(data_list[-1], self.optimizer, **kwargs)
        else:
            assert False
            # outputs = self.model.val_step(data_batch, self.optimizer, **kwargs)

        if not isinstance(outputs, dict):
            raise TypeError('"batch_processor()" or "model.train_step()"'
                            'and "model.val_step()" must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs

