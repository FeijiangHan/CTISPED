import numpy as np
import torch
import SimpleITK as sitk
import csv
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, LogFormatter, StrMethodFormatter, FixedFormatter
from scipy.spatial.distance import cdist

FROC_minX = 0.125  # Mininum value of x-axis of FROC curve
FROC_maxX = 16  # Maximum value of x-axis of FROC curve
bLogPlot = True


def compute_dist(loc_rib, loc_frac):
    x = torch.from_numpy(loc_rib).float()
    y = torch.from_numpy(loc_frac).float()

    x_norm = x * x
    y_norm = y * y

    matrix_d = x_norm.sum(1).unsqueeze(1) - 2 * torch.matmul(x, y.t()) + y_norm.sum(1).unsqueeze(0)

    min_dist, indices = matrix_d.min(dim=1)
    return min_dist.numpy(), indices.numpy()


def volume_nms(volume, prob_thresh=0.5, dist_thresh=2, max_num=20):
    volume[volume < prob_thresh] = 0
    volume_shape = volume.shape
    nms_loc = []
    nms_prob = []
    for i in range(max_num):
        if volume.max() < prob_thresh:
            break

        loc = np.unravel_index(np.argmax(volume, axis=None), volume.shape)
        prob = volume[loc]
        nms_loc.append(loc)
        nms_prob.append(prob)

        z_range = np.clip(np.array([loc[0] - dist_thresh, loc[0] + dist_thresh]), a_min=0, a_max=volume_shape[0] - 1)
        y_range = np.clip(np.array([loc[1] - dist_thresh, loc[1] + dist_thresh]), a_min=0, a_max=volume_shape[1] - 1)
        x_range = np.clip(np.array([loc[2] - dist_thresh, loc[2] + dist_thresh]), a_min=0, a_max=volume_shape[2] - 1)

        volume[z_range[0]:z_range[1] + 1, y_range[0]:y_range[1] + 1, x_range[0]:x_range[1] + 1] = 0

    return nms_loc, nms_prob


def NMS_W(loc, prob, dist_th=30, weight=[2, 1, 1]):
    selected = np.zeros(loc.shape[0])
    target_loc = []
    target_prob = []
    target_index = []
    while selected.sum() < selected.shape[0]:
        max_idx = np.argmax(prob * (selected < 0.5))
        target_index.append(max_idx)
        selected[max_idx] = 1
        keep_loc = loc[max_idx:max_idx + 1]

        target_prob.append(prob[max_idx])
        dist = cdist(loc, keep_loc, 'wminkowski', p=2, w=weight).squeeze()
        tmp_idx = dist < dist_th
        selected[tmp_idx] = 1
        # target_loc.append(loc[min_dist < dist_th*dist_th].mean(axis=0, keepdims=True))
        # target_loc.append(loc[max_idx:max_idx+1])

        target_loc.append(np.average(loc[tmp_idx], axis=0, weights=prob[tmp_idx]))
    target_index = np.array(target_index)
    target_loc = np.stack(target_loc, axis=0)
    target_prob = np.array(target_prob)

    return target_loc, target_prob, target_index


def non_maximum_suppression(loc, prob, dist_th=30):
    selected = np.zeros(loc.shape[0])
    target_loc = []
    target_prob = []

    while selected.sum() < selected.shape[0]:
        max_idx = np.argmax(prob * (selected < 0.5))
        selected[max_idx] = 1
        keep_loc = loc[max_idx:max_idx + 1]

        target_prob.append(prob[max_idx])
        min_dist, _ = compute_dist(loc, keep_loc)
        selected[min_dist < dist_th * dist_th] = 1
        target_loc.append(loc[min_dist < dist_th * dist_th].mean(axis=0, keepdims=True))

    target_loc = np.concatenate(target_loc, axis=0)
    target_prob = np.array(target_prob)

    return target_loc, target_prob


def non_maximum_suppression_weight(loc, prob, dist_th=30, weight=[2, 1, 1]):
    loc_weighted = np.concatenate((loc[:, 0:1] * weight[0], loc[:, 1:2] * weight[1], loc[:, 2:3] * weight[2]), axis=1)
    selected = np.zeros(loc.shape[0])
    target_loc = []
    target_prob = []

    while selected.sum() < selected.shape[0]:
        max_idx = np.argmax(prob * (selected < 0.5))
        selected[max_idx] = 1
        keep_loc = loc_weighted[max_idx:max_idx + 1]

        target_prob.append(prob[max_idx])
        min_dist, _ = compute_dist(loc_weighted, keep_loc)
        tmp_idx = min_dist < dist_th * dist_th
        selected[tmp_idx] = 1
        # target_loc.append(loc[min_dist < dist_th*dist_th].mean(axis=0, keepdims=True))
        # target_loc.append(loc[max_idx:max_idx+1])

        target_loc.append(np.average(loc[tmp_idx], axis=0, weights=prob[tmp_idx]))

    target_loc = np.stack(target_loc, axis=0)
    target_prob = np.array(target_prob)

    return target_loc, target_prob


def non_maximum_suppression_class(loc, prob, cls0, cls1, dist_th=30, weight=[2, 1, 1]):
    loc_weighted = np.concatenate((loc[:, 0:1] * weight[0], loc[:, 1:2] * weight[1], loc[:, 2:3] * weight[2]), axis=1)
    selected = np.zeros(loc.shape[0])
    target_loc = []
    target_prob = []
    target_cls0 = []
    target_cls1 = []

    while selected.sum() < selected.shape[0]:
        max_idx = np.argmax(prob * (selected < 0.5))
        selected[max_idx] = 1
        keep_loc = loc_weighted[max_idx:max_idx + 1]

        target_prob.append(prob[max_idx])
        target_cls0.append(cls0[max_idx])
        target_cls1.append(cls1[max_idx])
        min_dist, _ = compute_dist(loc_weighted, keep_loc)
        tmp_idx = min_dist < dist_th * dist_th
        selected[tmp_idx] = 1
        # target_loc.append(loc[min_dist < dist_th*dist_th].mean(axis=0, keepdims=True))
        # target_loc.append(loc[max_idx:max_idx+1])

        try:
            target_loc.append(np.average(loc[tmp_idx], axis=0, weights=prob[tmp_idx]))
        except:
            fail = 1

    target_loc = np.stack(target_loc, axis=0)
    target_prob = np.array(target_prob)
    target_cls0 = np.array(target_cls0)
    target_cls1 = np.array(target_cls1)

    return target_loc, target_prob, target_cls0, target_cls1
