#!/usr/bin/python3
# -*- coding: utf-8 -*
import torch
from torch import nn
from .network_blocks import DoubleConvBlock, AttentionBlock2d, ResidualRecurrentBlock


class UNet(nn.Module):
    """基本的UNet网络，其中的卷积并不会改变张量的尺寸"""

    def __init__(self, num_classes, in_channels, is_attention=False, is_recurrent_residual=False):
        super().__init__()
        nb_filters = [32, 64, 128, 256, 512]
        self.is_attention = is_attention
        self.is_recurrent_residual = is_recurrent_residual

        # 标准的2倍上采样和下采样，因为没有可以学习的参数，可以共享
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        if is_recurrent_residual:
            # 下采样的模块
            self.conv0_0 = ResidualRecurrentBlock(in_channels, nb_filters[0])
            self.conv1_0 = ResidualRecurrentBlock(nb_filters[0], nb_filters[1])
            self.conv2_0 = ResidualRecurrentBlock(nb_filters[1], nb_filters[2])
            self.conv3_0 = ResidualRecurrentBlock(nb_filters[2], nb_filters[3])
            self.conv4_0 = ResidualRecurrentBlock(nb_filters[3], nb_filters[4])

            # 上采样的模块
            self.conv3_1 = ResidualRecurrentBlock(nb_filters[4] + nb_filters[3], nb_filters[3])
            self.conv2_2 = ResidualRecurrentBlock(nb_filters[3] + nb_filters[2], nb_filters[2])
            self.conv1_3 = ResidualRecurrentBlock(nb_filters[2] + nb_filters[1], nb_filters[1])
            self.conv0_4 = ResidualRecurrentBlock(nb_filters[1] + nb_filters[0], nb_filters[0])
        else:
            # 下采样的模块
            self.conv0_0 = DoubleConvBlock(in_channels, nb_filters[0], nb_filters[0])
            self.conv1_0 = DoubleConvBlock(nb_filters[0], nb_filters[1], nb_filters[1])
            self.conv2_0 = DoubleConvBlock(nb_filters[1], nb_filters[2], nb_filters[2])
            self.conv3_0 = DoubleConvBlock(nb_filters[2], nb_filters[3], nb_filters[3])
            self.conv4_0 = DoubleConvBlock(nb_filters[3], nb_filters[4], nb_filters[4])

            # 上采样的模块
            self.conv3_1 = DoubleConvBlock(nb_filters[4] + nb_filters[3], nb_filters[3], nb_filters[3])
            self.conv2_2 = DoubleConvBlock(nb_filters[3] + nb_filters[2], nb_filters[2], nb_filters[2])
            self.conv1_3 = DoubleConvBlock(nb_filters[2] + nb_filters[1], nb_filters[1], nb_filters[1])
            self.conv0_4 = DoubleConvBlock(nb_filters[1] + nb_filters[0], nb_filters[0], nb_filters[0])

        # 最后接一个Conv计算在所有类别上的分数
        self.final = nn.Conv2d(nb_filters[0], num_classes, kernel_size=1, stride=1)

        if is_attention:
            self.attention3_1 = AttentionBlock2d(nb_filters[3], nb_filters[4])
            self.attention2_2 = AttentionBlock2d(nb_filters[2], nb_filters[3])
            self.attention1_3 = AttentionBlock2d(nb_filters[1], nb_filters[2])
            self.attention0_4 = AttentionBlock2d(nb_filters[0], nb_filters[1])

    def forward(self, input):
        # 下采样编码
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.down(x0_0))
        x2_0 = self.conv2_0(self.down(x1_0))
        x3_0 = self.conv3_0(self.down(x2_0))
        x4_0 = self.conv4_0(self.down(x3_0))

        if self.is_attention:
            # 特征融合并进行上采样解码，使用concatenate进行特征融合
            x3_1 = self.conv3_1(torch.cat([self.attention3_1(x3_0, x4_0), self.up(x4_0)], dim=1))
            x2_2 = self.conv2_2(torch.cat([self.attention2_2(x2_0, x3_1), self.up(x3_1)], dim=1))
            x1_3 = self.conv1_3(torch.cat([self.attention1_3(x1_0, x2_2), self.up(x2_2)], dim=1))
            x0_4 = self.conv0_4(torch.cat([self.attention0_4(x0_0, x1_3), self.up(x1_3)], dim=1))
        else:
            # 特征融合并进行上采样解码，使用concatenate进行特征融合
            x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], dim=1))
            x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], dim=1))
            x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], dim=1))
            x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], dim=1))

        # 计算每个类别上的得分
        output = self.final(x0_4)

        return output

    def __str__(self):
        ret = 'U-Net'
        if self.is_recurrent_residual:
            ret = 'R2U-Net'
        if self.is_attention:
            ret = 'Attention ' + ret
        return ret
