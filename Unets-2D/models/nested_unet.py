#!/usr/bin/python3
# -*- coding: utf-8 -*
from torch import nn
import torch
from .network_blocks import DoubleConvBlock


class NestedUNet(nn.Module):

    def __init__(self, num_classes, in_channels, is_deep_supervision=True, is_dense_connections=True, l_num=4):
        super().__init__()

        assert l_num in [2, 3, 4], "目前实现的UNet++只有4层"

        nb_filters = [32, 64, 128, 256, 512]

        self.is_deep_supervision = is_deep_supervision
        self.is_dense_connections = is_dense_connections
        self.l_num = l_num

        # 这2个层内没有需要学习的参数，可以共享
        # 使用MaxPooling进行2倍下采样
        self.downsampling = nn.MaxPool2d(kernel_size=2, stride=2)
        # 使用双线性插值进行2倍上采样，对于pixel-wise任务，align_corners设置为True
        self.upsampling = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = DoubleConvBlock(in_channels, nb_filters[0], nb_filters[0])
        self.conv1_0 = DoubleConvBlock(nb_filters[0], nb_filters[1], nb_filters[1])
        self.conv2_0 = DoubleConvBlock(nb_filters[1], nb_filters[2], nb_filters[2])
        self.conv3_0 = DoubleConvBlock(nb_filters[2], nb_filters[3], nb_filters[3])
        self.conv4_0 = DoubleConvBlock(nb_filters[3], nb_filters[4], nb_filters[4])

        self.conv0_1 = DoubleConvBlock(nb_filters[0] + nb_filters[1], nb_filters[0], nb_filters[0])
        self.conv1_1 = DoubleConvBlock(nb_filters[1] + nb_filters[2], nb_filters[1], nb_filters[1])
        self.conv2_1 = DoubleConvBlock(nb_filters[2] + nb_filters[3], nb_filters[2], nb_filters[2])
        self.conv3_1 = DoubleConvBlock(nb_filters[3] + nb_filters[4], nb_filters[3], nb_filters[3])

        # 是否使用dense connections
        if self.is_dense_connections:
            self.conv0_2 = DoubleConvBlock(nb_filters[0] * 2 + nb_filters[1], nb_filters[0], nb_filters[0])
            self.conv1_2 = DoubleConvBlock(nb_filters[1] * 2 + nb_filters[2], nb_filters[1], nb_filters[1])
            self.conv2_2 = DoubleConvBlock(nb_filters[2] * 2 + nb_filters[3], nb_filters[2], nb_filters[2])

            self.conv0_3 = DoubleConvBlock(nb_filters[0] * 3 + nb_filters[1], nb_filters[0], nb_filters[0])
            self.conv1_3 = DoubleConvBlock(nb_filters[1] * 3 + nb_filters[2], nb_filters[1], nb_filters[1])

            self.conv0_4 = DoubleConvBlock(nb_filters[0] * 4 + nb_filters[1], nb_filters[0], nb_filters[0])
        else:
            self.conv0_2 = DoubleConvBlock(nb_filters[0] + nb_filters[1], nb_filters[0], nb_filters[0])
            self.conv1_2 = DoubleConvBlock(nb_filters[1] + nb_filters[2], nb_filters[1], nb_filters[1])
            self.conv2_2 = DoubleConvBlock(nb_filters[2] + nb_filters[3], nb_filters[2], nb_filters[2])

            self.conv0_3 = DoubleConvBlock(nb_filters[0] + nb_filters[1], nb_filters[0], nb_filters[0])
            self.conv1_3 = DoubleConvBlock(nb_filters[1] + nb_filters[2], nb_filters[1], nb_filters[1])

            self.conv0_4 = DoubleConvBlock(nb_filters[0] + nb_filters[1], nb_filters[0], nb_filters[0])

        # 如果使用deep supervision，则每层UNet都有一个输出
        if self.is_deep_supervision:
            self.final1 = nn.Conv2d(nb_filters[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filters[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filters[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filters[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filters[0], num_classes, kernel_size=1)

        # 删除多余层的变量
        if self.l_num == 2:
            del self.conv3_0, self.conv2_1, self.conv1_2, self.conv0_3
            del self.conv4_0, self.conv3_1, self.conv2_2, self.conv1_3, self.conv0_4
        if self.l_num == 3:
            del self.conv4_0, self.conv3_1, self.conv2_2, self.conv1_3, self.conv0_4

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.downsampling(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.upsampling(x1_0)], dim=1))

        x2_0 = self.conv2_0(self.downsampling(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.upsampling(x2_0)], dim=1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.upsampling(x1_1)], dim=1))

        if self.l_num == 2:
            if self.is_deep_supervision:
                return [self.final1(x0_1), self.final2(x0_2)]
            else:
                return self.final(x0_2)

        x3_0 = self.conv3_0(self.downsampling(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.upsampling(x3_0)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.upsampling(x2_1)], dim=1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.upsampling(x1_2)], dim=1))

        if self.l_num == 3:
            if self.is_deep_supervision:
                return [self.final1(x0_1), self.final2(x0_2), self.final3(x0_3)]
            else:
                return self.final(x0_3)

        x4_0 = self.conv4_0(self.downsampling(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.upsampling(x4_0)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.upsampling(x3_1)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.upsampling(x2_2)], dim=1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.upsampling(x1_3)], dim=1))

        if self.is_deep_supervision:
            return [self.final1(x0_1), self.final2(x0_2), self.final3(x0_3), self.final4(x0_4)]
        else:
            return self.final(x0_4)

    def __str__(self):
        return 'UNet++(is_deep_supervision={}, is_dense_connections={}, l_num={})'.format(
            self.is_deep_supervision, self.is_dense_connections, self.l_num)
