import re
import functools
import torch
import torch.nn as nn
from utils import function as F
from .base import *
from torch.nn import functional


class JaccardLoss(Loss):

    def __init__(self, eps=1e-5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return 1 - F.jaccard(
            y_pr, y_gt,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
        )


class DiceLoss(Loss):

    def __init__(self, eps=1., beta=1., activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return 1 - F.f_score(
            y_pr, y_gt,
            beta=self.beta,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
        )


class FocalTverskyLoss(Loss):

    def __init__(self, eps=1., alpha=0.5, beta=0.5, square=True, batch_dice=True, drop_bg=False):
        super().__init__()
        self.eps = eps
        self.alpha = alpha
        self.beta = beta
        self.square = square
        self.drop_bg = drop_bg
        self.batch_dice = batch_dice
        if batch_dice:
            self.dim = [0, 1, 2, 3, 4]
        else:
            self.dim = [1, 2, 3, 4]

    def forward(self, pred, target):
        tp = torch.sum(pred * target, dim=self.dim)
        fp = torch.sum(pred ** 2 * (1. - target), dim=self.dim)
        fn = torch.sum((1 - pred) * target, dim=self.dim)

        tversky_score = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
        loss = 1. - tversky_score
        if not self.batch_dice and self.drop_bg:
            fg_idx = target.flatten(1).max(dim=1)[0] > 0
            if fg_idx.max() > 0:
                loss = loss[target.flatten(1).max(dim=1)[0] > 0].mean()
            else:
                loss = torch.tensor(0., device=pred.device)
        else:
            loss.mean()
        return loss


class FocalTverskyLoss2(Loss):

    def __init__(self, gamma=1., eps=1., alpha=0.5, beta=0.5, asymmetric=True, detach=False):
        super().__init__()
        self.gamma = gamma
        self.eps = eps
        self.alpha = alpha
        self.beta = beta
        self.asymmetric = asymmetric
        self.detach = detach

    def forward(self, pred, target):
        if self.detach:
            pred_weight = pred.detach()
        else:
            pred_weight = pred
        if self.asymmetric:
            focal_weight = torch.where(torch.eq(target, 1.), torch.ones_like(pred_weight), pred_weight)
        else:
            focal_weight = torch.where(torch.eq(target, 1.), 1. - pred_weight, pred_weight)
        focal_weight = torch.pow(focal_weight, self.gamma)

        tp = torch.sum(pred * target)
        fp = torch.sum(pred * (1. - target) * focal_weight)
        fn = torch.sum((1 - pred) * target * focal_weight)
        tversky_score = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
        loss = 1. - tversky_score

        return loss


class TverskyLoss(Loss):

    def __init__(self, eps=1., alpha=0.5, beta=0.5, square=True, batch_dice=True, drop_bg=False):
        super().__init__()
        self.eps = eps
        self.alpha = alpha
        self.beta = beta
        self.square = square
        self.drop_bg = drop_bg
        self.batch_dice = batch_dice
        if batch_dice:
            self.dim = [0, 2, 3, 4]
        else:
            self.dim = [2, 3, 4]

    def forward(self, pred, target):
        tp = torch.sum(pred * target, dim=self.dim)  # TP
        if self.square:
            fp = torch.sum(pred ** 2 * (1. - target), dim=self.dim)
            fn = torch.sum((1 - pred) ** 2 * target, dim=self.dim)
        else:
            fp = torch.sum(pred * (1. - target), dim=self.dim)
            fn = torch.sum((1 - pred) * target, dim=self.dim)
        tversky_score = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
        loss = 1. - tversky_score
        if not self.batch_dice and self.drop_bg:
            l = []
            for cls in range(target.shape[1]):
                fg_idx = target[:, cls].flatten(1).max(dim=1)[0] > 0
                if fg_idx.max() > 0:
                    l.append(loss[fg_idx, cls].mean())
                else:
                    l.append(torch.tensor(0., device=pred.device))
            loss = torch.stack(l).mean()
        else:
            loss = loss.mean()
        return loss


class GroupTverskyLoss(Loss):

    def __init__(self, eps=1., alpha=0.5, beta=0.5, square=True, group=1, drop_bg=False):
        super().__init__()
        self.eps = eps
        self.alpha = alpha
        self.beta = beta
        self.square = square
        self.drop_bg = drop_bg
        self.group = group

    def forward(self, pred, target):
        pred = pred.permute(1, 0, 2, 3, 4).flatten(1)  # C*NDHW
        target = target.permute(1, 0, 2, 3, 4).flatten(1)

        pred = pred.reshape([pred.shape[0], self.group, -1])  # C*N*DHW
        target = target.reshape([target.shape[0], self.group, -1])  # C*N*DHW

        tp = torch.sum(pred * target, dim=[2])  # TP
        if self.square:
            fp = torch.sum(pred ** 2 * (1. - target), dim=[2])
            fn = torch.sum((1 - pred) ** 2 * target, dim=[2])
        else:
            fp = torch.sum(pred * (1. - target), dim=[2])
            fn = torch.sum((1 - pred) * target, dim=[2])
        tversky_score = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
        loss = 1. - tversky_score
        if self.drop_bg:
            l = []
            for cls in range(target.shape[0]):
                fg_idx = target[cls].flatten(1).max(dim=1)[0] > 0
                if fg_idx.max() > 0:
                    l.append(loss[cls, fg_idx].mean())
                else:
                    l.append(torch.tensor(0., device=pred.device))
            loss = torch.stack(l).mean()
        else:
            loss = loss.mean()
        return loss


class TverskyLoss1(Loss):

    def __init__(self, eps=1., alpha=0.5, beta=0.5, square=True, batch_dice=True, drop_bg=False):
        super().__init__()
        self.eps = eps
        self.alpha = alpha
        self.beta = beta
        self.square = square
        self.drop_bg = drop_bg
        self.batch_dice = batch_dice
        if batch_dice:
            self.dim = [0, 1, 2, 3, 4]
        else:
            self.dim = [1, 2, 3, 4]

    def forward(self, pred, target):
        tp = torch.sum(pred * target, dim=self.dim)  # TP
        if self.square:
            fp = torch.sum(pred ** 2 * (1. - target), dim=self.dim)
            fn = torch.sum((1 - pred) ** 2 * target, dim=self.dim)
        else:
            fp = torch.sum(pred * (1. - target), dim=self.dim)
            fn = torch.sum((1 - pred) * target, dim=self.dim)
        tversky_score = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
        loss = 1. - tversky_score
        if not self.batch_dice and self.drop_bg:
            fg_idx = target.flatten(1).max(dim=1)[0] > 0
            if fg_idx.max() > 0:
                loss = loss[fg_idx].mean()
            else:
                loss = torch.tensor(0., device=pred.device)

        else:
            loss = loss.mean()
        return loss


class LogTverskyLoss(Loss):

    def __init__(self, eps=1., alpha=0.5, beta=0.5, gamma=0.3, square=True, batch_dice=True):
        super().__init__()
        self.eps = eps
        self.alpha = alpha
        self.beta = beta
        self.square = square
        self.batch_dice = batch_dice
        self.gamma = gamma
        if batch_dice:
            self.dim = [0, 1, 2, 3, 4]
        else:
            self.dim = [1, 2, 3, 4]

    def forward(self, pred, target):
        tp = torch.sum(pred * target, dim=self.dim)  # TP
        if self.square:
            fp = torch.sum(pred ** 2 * (1. - target), dim=self.dim)
            fn = torch.sum((1 - pred) ** 2 * target, dim=self.dim)
        else:
            fp = torch.sum(pred * (1. - target), dim=self.dim)
            fn = torch.sum((1 - pred) * target, dim=self.dim)
        tversky_score = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
        loss = torch.pow(-torch.log(torch.clamp(tversky_score, min=1e-4)), self.gamma)
        loss = loss.mean()
        return loss


class GeneralizedTverskyLoss(Loss):

    def __init__(self, eps=1., square=True):
        super().__init__()
        self.eps = eps
        self.square = square

    def forward(self, pred, target):
        target = torch.cat([target, 1 - target], dim=1)

        pred = pred.flatten(2)
        target = target.flatten(2)
        w_l = target.sum(-1).detach()
        w_l = 1. / w_l.clamp(min=1)

        tp = torch.sum(pred * target, dim=-1)  # TP
        if self.square:
            fp = torch.sum(pred ** 2 * (1. - target), dim=-1)
            fn = torch.sum((1 - pred) ** 2 * target, dim=-1)
        else:
            fp = torch.sum(pred * (1. - target), dim=-1)
            fn = torch.sum((1 - pred) * target, dim=-1)
        tversky_score = ((w_l * tp).sum(-1)) / ((w_l * (tp + 0.5 * fp + 0.5 * fn)).sum(-1))
        loss = 1. - tversky_score.mean()

        return loss


class FocalBCE(Loss):

    def __init__(self, gamma=2., alpha=0.5, beta=0.5, min_pixel=100, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.min_pixel = min_pixel

    def forward(self, pred, target):
        batch_size = pred.shape[0]
        alpha = torch.tensor(self.alpha, device=pred.device)
        beta = torch.tensor(self.beta, device=pred.device)
        alpha_factor = torch.where(torch.eq(target, 1.), alpha, beta)
        focal_weight = torch.where(torch.eq(target, 1.), 1. - pred, pred)
        focal_weight = torch.pow(focal_weight, self.gamma) * alpha_factor
        bce = functional.binary_cross_entropy(pred, target, reduction='none')
        loss = (bce * focal_weight).sum() / torch.clamp(target.sum(), min=batch_size * self.min_pixel)
        return loss


class OHEM(Loss):

    def __init__(self, mult=5, min_pixel=1000, pos_weight=2, **kwargs):
        super().__init__(**kwargs)
        self.mult = mult
        self.min_pixel = min_pixel
        self.pos_weight = pos_weight

    def forward(self, pred, target):
        batch_size = target.shape[0]
        pred = pred.flatten(1)
        target = target.flatten(1)
        weight = torch.where(target == 1, torch.ones_like(target) * self.pos_weight, torch.ones_like(target))
        bce = functional.binary_cross_entropy(pred, target, weight=weight, reduction='none')

        loss = []
        for i in range(batch_size):
            pos_num = torch.sum(target[i]).int().item()
            pos_bce = bce[i, (target[i] == 1)]
            neg_bce = bce[i, (target[i] == 0)]
            neg_bce = torch.topk(neg_bce, max(pos_num * self.mult, self.min_pixel))[0]

            loss.append(torch.cat([pos_bce, neg_bce]).mean())

        loss = torch.stack(loss).mean()
        return loss


class WeightedBCE(Loss):

    def __init__(self, alpha=1, beta=1, eps=1e-3):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, pred, target):
        pred = torch.clamp(pred, min=self.eps, max=1 - self.eps)
        alpha = torch.tensor(self.alpha, device=pred.device).float()
        beta = torch.tensor(self.beta, device=pred.device).float()
        weight = torch.where(torch.eq(target, 1.), alpha, beta)
        bce = functional.binary_cross_entropy(pred, target, reduction='none')
        loss = (bce * weight).mean()
        return loss


class L1Loss(nn.L1Loss, Loss):
    pass


class MSELoss(nn.MSELoss, Loss):
    pass


class CrossEntropyLoss(nn.CrossEntropyLoss, Loss):
    pass


class NLLLoss(nn.NLLLoss, Loss):
    pass


class BCELoss(nn.BCELoss, Loss):
    pass


class BCEWithLogitsLoss(nn.BCEWithLogitsLoss, Loss):
    pass


class SmoothL1Loss(nn.SmoothL1Loss, Loss):
    pass
