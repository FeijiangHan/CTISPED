from .base import Loss
from .utils import *
import torch.nn.functional as F


class SegLoss(Loss):
    def __init__(self, result_key, sample_key, result_down, label_channel, seg_loss, loss_weight, reduction='sum'):
        super(SegLoss, self).__init__()
        self.result_key = result_key
        self.sample_key = sample_key
        self.result_down = result_down
        self.seg_loss = seg_loss
        self.loss_weight = loss_weight
        self.label_channel = label_channel
        self.reduction = reduction

    def forward(self, result_dict, sample_dict):
        loss = []
        for result_key, sample_key, down_rate, label_channel, seg_loss, loss_weight in zip(self.result_key,
                                                                                           self.sample_key,
                                                                                           self.result_down,
                                                                                           self.label_channel,
                                                                                           self.seg_loss,
                                                                                           self.loss_weight):
            if down_rate > 1:
                label_down = F.max_pool3d(sample_dict[sample_key][:, label_channel], stride=down_rate,
                                          kernel_size=down_rate)
                pred = result_dict[result_key]
                loss.append(seg_loss(pred, label_down) * loss_weight)
            else:
                pred = result_dict[result_key]
                loss.append(seg_loss(pred, sample_dict[sample_key][:, label_channel]) * loss_weight)
        if self.reduction == 'mean':
            loss = torch.stack(loss).mean()
        elif self.reduction == 'sum':
            loss = torch.stack(loss).sum()

        return loss
