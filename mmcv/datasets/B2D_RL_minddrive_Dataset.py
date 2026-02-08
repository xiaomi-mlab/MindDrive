# ------------------------------------------------------------------------
# Modified from Bench2Drive(https://github.com/Thinklab-SJTU/Bench2Drive)
# Copyright (c) Xiaomi Corporation. All rights reserved.
# ------------------------------------------------------------------------
from torch.utils.data import Dataset
import os
import numpy as np
import mmcv
from mmcv.datasets import DATASETS
import torch
from mmcv.fileio.parse import list_from_file
import glob
import pickle

@DATASETS.register_module()
class RL_minddrive_Dataset(Dataset):
    def __init__(self, data_root, classes=None, index_file=None):
        super().__init__()
        with open(data_root, 'rb') as f:
            self.index_paths = pickle.load(f)

        self.length = len(self.index_paths)
        self.CLASSES = self.get_classes(classes)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        path = self.index_paths[idx]
        data = np.load(path, allow_pickle=True)


        sample = {
            "actions": torch.from_numpy(data["actions"][0]).float(),
            "rewards": torch.from_numpy(data["rewards"][0]).float(),
            "returns": torch.from_numpy(data["returns"][0]).float(),
            "values": torch.from_numpy(data["values"][0]).float(),
            "advantages": torch.from_numpy(data["advantages"][0]).float(),
            "ref_log_probs": torch.from_numpy(data["ref_log_probs"][0]).float(),

        }
        inputs_embeds = data['meta_action_info'][0][0]['inputs_embeds'][0]
        new_input_ids = data['meta_action_info'][0][0]['new_input_ids'][0]
        sample.update({
            'inputs_embeds': inputs_embeds,
            'new_input_ids': new_input_ids
        })

        return sample

    @classmethod
    def get_classes(cls, classes=None):
        if classes is None:
            return getattr(cls, "CLASSES", None)
        if isinstance(classes, str):
            class_names = list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')
        return class_names