#!/usr/bin/python3
# -*- coding: utf-8 -*
from models.lovasz_losses import lovasz_softmax
from torch import nn
from torch.nn import functional as F
import torch


class LovaszLoss(nn.Module):
    """lovasz损失函数，https://github.com/bermanmaxim/LovaszSoftmax"""

    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        probability = F.softmax(input, dim=1)
        loss = lovasz_softmax(probability, target)
        return loss


class SoftDiceLoss(nn.Module):
    """Dice损失函数，仅用于二分类，https://www.aiuai.cn/aifarm1159.html，https://github.com/pytorch/pytorch/issues/1249"""

    def __init__(self, num_classes, smooth=1):
        """:arg smooth Dice系数分母分子上添加的参数，防止除0错误"""

        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, input, target):
        # 确保SoftDiceLoss只用于二分类问题
        assert torch.max(target).item() <= 1, 'SoftDiceLoss()目前只能用于二分类'

        batch_size = input.size(0)
        probability = F.softmax(input, dim=1)

        # 将probability和target矩阵flatten为B*(C*H*W)的矩阵
        probability = probability.view(batch_size, -1)
        # 这里的target label需要是one hot格式，one hot要显示传入类别数量
        target_one_hot = F.one_hot(target, num_classes=self.num_classes).permute((0, 3, 1, 2))
        target_one_hot = target_one_hot.contiguous().view(batch_size, -1)
        # 计算相交区域的概率矩阵
        intersection = probability * target_one_hot

        # 计算每一个batch的Dice相似系数（Dice similar coefficient）
        dsc = (2 * intersection.sum(dim=1) + self.smooth) / (
                target_one_hot.sum(dim=1) + probability.sum(dim=1) + self.smooth)
        loss = (1 - dsc).sum()

        return loss


# class BCELoss2d(nn.Module):
#     """二进制交叉熵，计算概率的函数可选Sigmoid和Softmax，https://www.aiuai.cn/aifarm1159.html"""
#
#     def __init__(self, p_function='sigmoid', weight=None, size_average=True):
#         super().__init__()
#
#         assert p_function in ['softmax', 'sigmoid']
#
#         self.p_function = p_function
#         self.bce_loss_function = nn.BCELoss(weight=weight, size_average=size_average)
#
#     def forward(self, input, target):
#         if self.p_function == 'softmax':
#             probability = F.softmax(input, dim=1)
#         else:
#             probability = F.sigmoid(input)
#         loss = self.bce_loss_function(probability, target)
#         return loss

class DiceAndBCELoss(nn.Module):
    """Dice损失和BCE损失的混合"""

    def __init__(self, num_classes):
        super().__init__()
        self.dice_loss = SoftDiceLoss(num_classes=num_classes)
        self.bce_loss = nn.CrossEntropyLoss()

    def forward(self, input, target):
        return self.dice_loss(input, target) + self.bce_loss(input, target)


class SCELoss(nn.Module):
    """用于Learning with noisy labels的对称交叉熵损失函数"""

    def __init__(self, num_classes, alpha=1.0, beta=1.0):
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, input, target):
        # CCE
        ce = self.cross_entropy(input, target)

        # RCE
        input = F.softmax(input, dim=1)
        input = torch.clamp(input, min=1e-7, max=1.0)
        target_one_hot = F.one_hot(target, self.num_classes).permute((0, 3, 1, 2)).float()
        target_one_hot = torch.clamp(target_one_hot, min=1e-4, max=1.0)
        rce = (-1 * torch.sum(input * torch.log(target_one_hot), dim=1))

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss

    def __str__(self):
        return 'SCELoss(alpha={}, beta={})'.format(self.alpha, self.beta)
