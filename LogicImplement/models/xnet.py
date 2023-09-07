#!/usr/bin/python3
# -*- coding: utf-8 -*
import torch
from torch import nn
from .network_blocks import SingleConvBlock


class XNet(nn.Module):
    """用于数据量小的X-Ray图像分割的网络，架构上是将2个SegNet拼接了起来，并添加了一些skip connection，
    https://github.com/JosephPB/XNet，https://arxiv.org/abs/1812.00548
    """

    def __init__(self, num_classes, in_channels):
        super().__init__()
        # TODO: nb_filters需要确认下
        nb_filters = [32, 64, 128, 256]

        # 标准的2倍上采样和下采样，因为没有可以学习的参数，可以共享
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        # 第一个SegNet
        self.conv0 = SingleConvBlock(in_channels, nb_filters[0])
        self.conv1 = SingleConvBlock(nb_filters[0], nb_filters[1])
        self.conv2 = SingleConvBlock(nb_filters[1], nb_filters[2])
        self.conv3 = nn.Sequential(
            SingleConvBlock(nb_filters[2], nb_filters[3]),
            SingleConvBlock(nb_filters[3], nb_filters[3])
        )
        self.conv4 = SingleConvBlock(nb_filters[3] + nb_filters[2], nb_filters[2])
        self.conv5 = SingleConvBlock(nb_filters[2] + nb_filters[1], nb_filters[1])

        # 第2个SegNet
        self.conv6 = SingleConvBlock(nb_filters[1], nb_filters[1])
        self.conv7 = SingleConvBlock(nb_filters[1], nb_filters[2])
        self.conv8 = nn.Sequential(
            SingleConvBlock(nb_filters[2], nb_filters[3]),
            SingleConvBlock(nb_filters[3], nb_filters[3])
        )
        self.conv9 = SingleConvBlock(nb_filters[3] + nb_filters[2], nb_filters[2])
        self.conv10 = SingleConvBlock(nb_filters[2] + nb_filters[1], nb_filters[1])
        self.conv11 = SingleConvBlock(nb_filters[1] + nb_filters[0], nb_filters[0])

        # 最后接一个Conv计算在所有类别上的分数
        self.final = nn.Conv2d(nb_filters[0], num_classes, kernel_size=1, stride=1)

    def forward(self, input):
        # 第一个SegNet
        x0 = self.conv0(input)
        x1 = self.conv1(self.down(x0))
        x2 = self.conv2(self.down(x1))
        x3 = self.conv3(self.down(x2))
        x4 = self.conv4(torch.cat([x2, self.up(x3)], dim=1))
        x5 = self.conv5(torch.cat([x1, self.up(x4)], dim=1))

        # 第2个SegNet
        x6 = self.conv6(x5)
        x7 = self.conv7(self.down(x6))
        x8 = self.conv8(self.down(x7))
        x9 = self.conv9(torch.cat([x7, self.up(x8)], dim=1))
        x10 = self.conv10(torch.cat([x6, self.up(x9)], dim=1))
        x11 = self.conv11(torch.cat([x0, self.up(x10)], dim=1))

        # 计算每个类别上的得分
        output = self.final(x11)

        return output

    def __str__(self):
        return 'X-Net'
