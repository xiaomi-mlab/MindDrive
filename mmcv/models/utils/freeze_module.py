import torch.nn as nn

def freeze_parameters(module:nn.Module):
    for name,param in module.named_parameters():
        param.requires_grad = False
        # print(f'freeze parm:{name}')

def freeze_batchnorm(module:nn.Module):
    for name, m in module.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
            m.eval()
            # print(f'freeze BN:{name}')


def freeze_module(module):
    freeze_parameters(module)
    freeze_batchnorm(module)
