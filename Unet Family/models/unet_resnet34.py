#!/usr/bin/python3
# -*- coding: utf-8 -*
# 基于ResNet34实现的UNet，将ResNet34作为UNet的Encoder模块
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models


class _DecoderBlock(nn.Module):
    """UNet中的解码模块，标准UNet中的Decoder模块"""

    def __init__(self, in_channels, middle_channels, out_channels=None):
        super().__init__()

        # 进行2次卷积后，再进行上采样（除了最后的输出的那一层）
        layers = [
            nn.Conv2d(in_channels, middle_channels, kernel_size=3),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, kernel_size=3),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True)
        ]

        if out_channels:
            # 最后进行2倍上采样
            layers.append(nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2))

        self.decode = nn.Sequential(*layers)

    def forward(self, input):
        return self.decode(input)


class UNet_ResNet34(nn.Module):
    """使用预训练的ResNet34（整体上分为4个stage，和UNet的下采样过程相似）作为Encoder实现的UNet"""

    def __init__(self, num_classes):
        super().__init__()

        # 加载预训练的ResNet34
        resnet34 = models.resnet34(pretrained=True)

        self.enc1 = nn.Sequential(
            resnet34.conv1,
            resnet34.bn1,
            resnet34.relu,
            resnet34.maxpool,
            # ResNet34中的layer1并不会进行下采样，所以将其和前面的层合并到一起
            resnet34.layer1
        )
        # 使用ResNet34中主要的4个stage最为UNet的Encoder层
        self.enc2 = resnet34.layer2
        self.enc3 = resnet34.layer3
        self.enc4 = resnet34.layer4

        self.center = _DecoderBlock(512, 1024, 512)

        self.dec4 = _DecoderBlock(1024, 512, 256)
        self.dec3 = _DecoderBlock(512, 256, 128)
        self.dec2 = _DecoderBlock(256, 128, 64)
        # 最后一层不进行上采样与降维
        self.dec1 = _DecoderBlock(128, 64)

        # 获得每个像素，在每个类别上的得分
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, input):
        enc1 = self.enc1(input)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)

        center = self.center(enc4)

        # 进行特征融合与上采样
        dec4 = self.dec4(self.feature_fusion(enc4, center))
        dec3 = self.dec3(self.feature_fusion(enc3, dec4))
        dec2 = self.dec2(self.feature_fusion(enc2, dec3))
        dec1 = self.dec1(self.feature_fusion(enc1, dec2))

        # 计算每个像素在每个类别上的得分
        final = self.final(dec1)

        # 最后通过上采样，将feature map还原到输入图像的尺寸
        return F.upsample_bilinear(final, input.shape[2:])

    def feature_fusion(self, t1, t2):
        """特征融合，将其中的一个张量进行下采样后（使其与另一个张量保持相同的shape），与另一个张量进行拼接"""

        # 假设输入的张量都是正方形的
        if t1.shape[-1] >= t2.shape[-1]:
            max_t = t1
            min_t = t2
        else:
            max_t = t2
            min_t = t1

        return torch.cat([F.upsample_bilinear(max_t, min_t.shape[2:]), min_t], dim=1)
