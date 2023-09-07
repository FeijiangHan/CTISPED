#!/usr/bin/python3
# -*- coding: utf-8 -*
import torch
from torch import nn
from .network_blocks import DoubleConvBlock


class OutputBlock(nn.Module):
    """论文中提到的输出块"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, input):
        return self.block(input)


class NFNPlus(nn.Module):
    """具有2个Stage的UNet结构的网络，添加了UNet间的skip connection，
    https://www.sciencedirect.com/science/article/pii/S0893608020300721
    """

    def __init__(self, num_classes, in_channels, is_deep_supervision=True, p=0.2):
        super().__init__()
        nb_filters = [32, 64, 128]
        self.is_deep_supervision = is_deep_supervision

        # 标准的2倍上采样和下采样，因为没有可以学习的参数，可以共享
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        # Front Network
        self.conv0_0 = DoubleConvBlock(in_channels, nb_filters[0], nb_filters[0], dropout=True, p=p)
        self.conv0_1 = DoubleConvBlock(nb_filters[0], nb_filters[1], nb_filters[1], dropout=True, p=p)
        self.conv0_2 = DoubleConvBlock(nb_filters[1], nb_filters[2], nb_filters[2], dropout=True, p=p)
        self.conv0_3 = DoubleConvBlock(nb_filters[2] + nb_filters[1], nb_filters[1], nb_filters[1], dropout=True, p=p)
        self.conv0_4 = DoubleConvBlock(nb_filters[1] + nb_filters[0], nb_filters[0], nb_filters[0], dropout=True, p=p)

        # Followed Network
        self.conv1_0 = DoubleConvBlock(nb_filters[0], nb_filters[0], nb_filters[0], dropout=True, p=p)
        self.conv1_1 = DoubleConvBlock(nb_filters[0] + nb_filters[1], nb_filters[1], nb_filters[1], dropout=True, p=p)
        self.conv1_2 = DoubleConvBlock(nb_filters[1] + nb_filters[2], nb_filters[2], nb_filters[2], dropout=True, p=p)
        self.conv1_3 = DoubleConvBlock(nb_filters[2] + nb_filters[1], nb_filters[1], nb_filters[1], dropout=True, p=p)
        self.conv1_4 = DoubleConvBlock(nb_filters[1] + nb_filters[0], nb_filters[0], nb_filters[0], dropout=True, p=p)

        if is_deep_supervision:
            self.final0 = OutputBlock(nb_filters[0], num_classes)
            self.final1 = OutputBlock(nb_filters[0], num_classes)
        else:
            self.final = OutputBlock(nb_filters[0], num_classes)

    def forward(self, input):
        # Front Network
        x0_0 = self.conv0_0(input)
        x0_1 = self.conv0_1(self.down(x0_0))
        x0_2 = self.conv0_2(self.down(x0_1))
        x0_3 = self.conv0_3(torch.cat([x0_1, self.up(x0_2)], dim=1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x0_3)], dim=1))

        # Followed Network
        x1_0 = self.conv1_0(x0_4)
        x1_1 = self.conv1_1(torch.cat([x0_3, self.down(x1_0)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x0_2, self.down(x1_1)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_1, self.up(x1_2)], dim=1))
        x1_4 = self.conv1_4(torch.cat([x1_0, self.up(x1_3)], dim=1))

        if self.is_deep_supervision:
            return [self.final0(x0_4), self.final1(x1_4)]
        else:
            return self.final(x1_4)

    def __str__(self):
        return 'NFN+'
