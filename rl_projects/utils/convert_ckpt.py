import pickle
import copy
import torch
import numpy as np

pth_file = '/high_perf_store3/data-lm/lf/train/od_ld_base/iter_100000.pth'

ckpt = torch.load(pth_file)

new_ckpt = copy.deepcopy(ckpt)
new_ckpt['state_dict'] = {}
for k, v in ckpt['state_dict'].items():
    if k.startswith('od.'):
        new_k = k[3:]
        new_ckpt['state_dict'][new_k] = v
    elif k.startswith('ld.') and 'ld.img_backbone' not in k:
        new_k = k[3:]
        new_ckpt['state_dict'][new_k] = v
torch.save(new_ckpt, '/high_perf_store3/data-lm/lf/train/od_ld_base/od_ld_convert_10w.pth')