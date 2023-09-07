import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from .panchors import shift


def track_running_stats(model):
    for m in model.modules():
        if isinstance(m, (_BatchNorm)):
            m.track_running_stats = True


def no_track_running_stats(model):
    for m in model.modules():
        if isinstance(m, (_BatchNorm)):
            m.track_running_stats = False


def calc_dst_3d(a, b):
    dist = torch.norm(torch.unsqueeze(b, dim=0) - torch.unsqueeze(a, dim=1), p=2, dim=-1)

    return dist


def calc_iou_3d(a, b):
    area = (b[:, 3] - b[:, 0]) * (b[:, 4] - b[:, 1]) * (b[:, 5] - b[:, 2])

    iz = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    iy = torch.min(torch.unsqueeze(a[:, 4], dim=1), b[:, 4]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])
    ix = torch.min(torch.unsqueeze(a[:, 5], dim=1), b[:, 5]) - torch.max(torch.unsqueeze(a[:, 2], 1), b[:, 2])

    iz = torch.clamp(iz, min=0)
    iy = torch.clamp(iy, min=0)
    ix = torch.clamp(ix, min=0)

    intersection = iz * iy * ix

    ua = torch.unsqueeze((a[:, 3] - a[:, 0]) * (a[:, 4] - a[:, 1]) * (a[:, 5] - a[:, 2]), dim=1) + area - intersection
    ua = torch.clamp(ua, min=1e-8)

    IoU = intersection.type(torch.float32) / ua.type(torch.float32)

    return IoU


def panchor_weight(shape, panchors, side=32, side_start=0):
    DEVICE = panchors.device
    side_dist1 = panchors - torch.tensor([0., 0., 0.], device=DEVICE).reshape(1, 1, 3)
    side_dist2 = torch.tensor(shape, device=DEVICE).float().reshape(1, 1, 3) - panchors

    side_dist = torch.min(torch.cat([side_dist1, side_dist2], dim=2), dim=2)[0]
    weight = torch.clamp(side_dist, max=side, min=side_start)
    weight = (weight - side_start) / (side - side_start)

    return weight


def ensemble_seg(seg, panchors, weights):
    min_total = torch.min(panchors.reshape(-1, 3), dim=0)[0]
    max_total = torch.max(panchors.reshape(-1, 3), dim=0)[0]

    total_shape = (max_total - min_total).int() + 1

    total_seg = torch.zeros([total_shape[0], total_shape[1], total_shape[2], 1]).to(seg.device)
    total_weights = torch.zeros([total_shape[0], total_shape[1], total_shape[2], 1]).to(seg.device)

    for i in range(seg.shape[0]):
        min_panchor = torch.min(panchors[i].reshape(-1, 3), dim=0)[0]
        max_panchor = torch.max(panchors[i].reshape(-1, 3), dim=0)[0]

        min_idx = (min_panchor - min_total).int()
        max_idx = (max_panchor - min_total).int() + 1

        total_seg[min_idx[0]:max_idx[0], min_idx[1]:max_idx[1], min_idx[2]:max_idx[2]] += seg[i] * weights[i]
        total_weights[min_idx[0]:max_idx[0], min_idx[1]:max_idx[1], min_idx[2]:max_idx[2]] += weights[i]

    total_seg = total_seg / (total_weights + 1e-6)

    return total_seg
