from . import base
from .image_process import show_box3
import numpy as np
import torch


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


class RecallW(base.Metric):

    def __init__(self, iou_thresh=0.2, prob_thresh=0.5):
        super().__init__()
        self.iou_thresh = iou_thresh
        self.prob_thresh = prob_thresh

    def forward(self, pr_box, pr_prob, gt_box):

        TP = 0
        gt_P = 0
        pr_P = 0

        for b in range(len(gt_box)):
            gt_P = gt_P + len(gt_box[b])
            pr_box_valid = pr_box[b][pr_prob[b] >= self.prob_thresh]

            if pr_box_valid.shape[0] == 0:
                continue
            if gt_box[b].shape[0] == 0:
                pr_P = pr_P + len(pr_box_valid)
                continue

            iou = calc_iou_3d(pr_box_valid, gt_box[b])
            TP += (iou.max(dim=0)[0] > self.iou_thresh).sum().item()

        recall = (TP + 1e-7) / (gt_P + 1e-7)

        return recall


class FPW(base.Metric):

    def __init__(self, iou_thresh=0.2, prob_thresh=0.5):
        super().__init__()
        self.iou_thresh = iou_thresh
        self.prob_thresh = prob_thresh

    def forward(self, pr_box, pr_prob, gt_box):

        TP = 0
        gt_P = 0
        pr_P = 0

        for b in range(len(gt_box)):
            gt_P = gt_P + len(gt_box[b])
            pr_box_valid = pr_box[b][pr_prob[b] >= self.prob_thresh]

            if pr_box_valid.shape[0] == 0:
                continue
            if gt_box[b].shape[0] == 0:
                pr_P = pr_P + len(pr_box_valid)
                continue

            iou = calc_iou_3d(pr_box_valid, gt_box[b])
            TP += (iou.max(dim=0)[0] > self.iou_thresh).sum().item()
            pr_P = pr_P + len(pr_box_valid)

        FP = (pr_P - TP) / len(gt_box)

        return FP


class PrecisionW(base.Metric):

    def __init__(self, iou_thresh=0.2, prob_thresh=0.5):
        super().__init__()
        self.iou_thresh = iou_thresh
        self.prob_thresh = prob_thresh

    def forward(self, pr_box, pr_prob, gt_box):

        TP = 0
        gt_P = 0
        pr_P = 0

        for b in range(len(gt_box)):
            gt_P = gt_P + len(gt_box[b])
            pr_box_valid = pr_box[b][pr_prob[b] >= self.prob_thresh]

            if pr_box_valid.shape[0] == 0:
                continue
            if gt_box[b].shape[0] == 0:
                pr_P = pr_P + len(pr_box_valid)
                continue

            iou = calc_iou_3d(pr_box_valid, gt_box[b])
            TP += (iou.max(dim=0)[0] > self.iou_thresh).sum().item()
            pr_P = pr_P + len(pr_box_valid)

        precision = (TP + 1e-7) / (pr_P + 1e-7)

        return precision


class FScoreW(base.Metric):

    def __init__(self, iou_thresh=0.2, prob_thresh=0.5, beta=1):
        super().__init__()
        self.iou_thresh = iou_thresh
        self.prob_thresh = prob_thresh
        self.beta = beta

    def forward(self, pr_box, pr_prob, gt_box):

        TP = 0
        gt_P = 0
        pr_P = 0

        for b in range(len(gt_box)):
            gt_P = gt_P + len(gt_box[b])
            pr_box_valid = pr_box[b][pr_prob[b] >= self.prob_thresh]

            if pr_box_valid.shape[0] == 0:
                continue
            if gt_box[b].shape[0] == 0:
                pr_P = pr_P + len(pr_box_valid)
                continue

            iou = calc_iou_3d(pr_box_valid, gt_box[b])
            TP += (iou.max(dim=0)[0] > self.iou_thresh).sum().item()
            pr_P = pr_P + len(pr_box_valid)

        precision = (TP + 1e-7) / (pr_P + 1e-7)
        recall = (TP + 1e-7) / (gt_P + 1e-7)
        fscore = ((1 + self.beta ** 2) * precision * recall) / ((self.beta ** 2) * precision + recall + 1e-7)
        return fscore
