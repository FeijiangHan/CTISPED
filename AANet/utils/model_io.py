from collections import OrderedDict
import torch
import os
import torch.nn as nn

version = torch.__version__


def DeParrallel(state_dict, exclude_name='Gooddd#'):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '')  # remove 'module.' of dataparallel
        if exclude_name not in name:
            new_state_dict[name] = v

    return new_state_dict


def ToSparseDict(state_dict, spconv_version):
    new_state_dict = OrderedDict()
    if spconv_version == 1:
        for name, tensor in state_dict.items():
            if 'up' in name and len(tensor.shape) == 5:  # transposedconv
                new_state_dict[name] = tensor.permute(2, 3, 4, 0, 1)
            elif len(tensor.shape) == 5:  # conv
                new_state_dict[name] = tensor.permute(2, 3, 4, 1, 0)
            elif 'norm' in name:
                new_state_dict[name.replace('norm', 'norm.norm')] = tensor
            else:
                new_state_dict[name] = tensor
    if spconv_version == 2:
        for name, tensor in state_dict.items():
            if 'up' in name and len(tensor.shape) == 5:  # transposedconv
                new_state_dict[name] = tensor.permute(1, 2, 3, 4, 0)
            elif len(tensor.shape) == 5:  # conv
                new_state_dict[name] = tensor.permute(0, 2, 3, 4, 1)
            elif 'norm' in name:
                new_state_dict[name.replace('norm', 'norm.norm')] = tensor
            else:
                new_state_dict[name] = tensor

    return new_state_dict


def SparseDictV2V1(state_dict):
    new_state_dict = OrderedDict()
    for name, tensor in state_dict.items():
        if 'up' in name and len(tensor.shape) == 5:  # transposedconv
            new_state_dict[name] = tensor.permute(4, 0, 1, 2, 3)
        elif len(tensor.shape) == 5:  # conv
            new_state_dict[name] = tensor.permute(4, 0, 1, 2, 3)
        else:
            new_state_dict[name] = tensor

    return new_state_dict


def Save(state_dict, path):
    state_dict = DeParrallel(state_dict, '###')
    if version[2] > '5':
        torch.save(state_dict, path, _use_new_zipfile_serialization=False)
    else:
        torch.save(state_dict, path)


def patch_first_conv(model, new_in_channels, default_in_channels=1, pretrained=True):
    """Change first convolution layer input channels.
    In case:
        in_channels == 1 or in_channels == 2 -> reuse original weights
        in_channels > 3 -> make random kaiming normal initialization
    """

    # get first conv
    for module in model.modules():
        if isinstance(module, nn.Conv3d) and module.in_channels == default_in_channels:
            break

    weight = module.weight.detach()
    module.in_channels = new_in_channels

    if not pretrained:
        module.weight = nn.parameter.Parameter(
            torch.Tensor(
                module.out_channels,
                new_in_channels // module.groups,
                *module.kernel_size
            )
        )
        module.reset_parameters()

    elif new_in_channels == 1:
        new_weight = weight.sum(1, keepdim=True)
        module.weight = nn.parameter.Parameter(new_weight)

    else:
        new_weight = torch.Tensor(
            module.out_channels,
            new_in_channels // module.groups,
            *module.kernel_size
        )

        for i in range(new_in_channels):
            new_weight[:, i] = weight[:, i % default_in_channels]

        new_weight = new_weight * (default_in_channels / new_in_channels)
        module.weight = nn.parameter.Parameter(new_weight)
