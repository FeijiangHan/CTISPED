#!/usr/bin/python3
# -*- coding: utf-8 -*
# 所用到的所有网络的基本模块
from torch import nn
from torch.nn import functional as F


class SingleConvBlock(nn.Module):
    """Conv-BN-ReLU的基本模块，其中的Conv并不会张量的尺寸"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.block(input)


class DoubleConvBlock(nn.Module):
    """由2个Conv-BN-ReLu组成的基本模块"""

    def __init__(self, in_channels, middle_channels, out_channels, dropout=False, p=0.2):
        super().__init__()
        # layers = [
        #     SingleConvBlock(in_channels, middle_channels),
        #     SingleConvBlock(middle_channels, out_channels)
        # ]
        #
        # if dropout:
        #     layers.append(nn.Dropout())

        # 将Dropout加在2个卷积层的中间
        layers = [SingleConvBlock(in_channels, middle_channels)]
        if dropout:
            layers.append(nn.Dropout(p=p))
        layers.append(SingleConvBlock(middle_channels, out_channels))

        self.block = nn.Sequential(*layers)

    def forward(self, input):
        return self.block(input)


class RecurrentSingleBlock(nn.Module):
    """具有循环结构的SingleBlock，其in_channels=out_channels"""

    def __init__(self, channels, t):
        assert t > 1, "循环次数必须>1"
        super().__init__()
        self.t = t
        self.block = SingleConvBlock(channels, channels)

    def forward(self, input):
        for i in range(self.t):
            if i == 0:
                output = self.block(input)
            # 循环时，要加上最开始的输入
            output = self.block(output + input)
        return output


class ResidualRecurrentBlock(nn.Module):
    """由2个RecurrentBlock和残差连接组成的模块"""

    def __init__(self, in_channels, out_channels, t=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.r2_block = nn.Sequential(
            # 在循环模块前加一个Conv用于改变数据的channel数
            RecurrentSingleBlock(out_channels, t),
            RecurrentSingleBlock(out_channels, t)
        )

    def forward(self, input):
        # 残差连接
        conv_input = self.conv(input)
        return self.r2_block(conv_input) + conv_input


class AttentionBlock2d(nn.Module):
    """注意力模块，计算较粗糙、低语义的feature map中每个点的注意力系数，并返回乘以注意力系数的feature map，
    按照论文作者的代码进行实现的
    https://github.com/ozan-oktay/Attention-Gated-Networks/blob/master/models/layers/grid_attention_layer.py"""

    def __init__(self, x_channels, gate_channels, inter_channels=None, subsample_factor=(2, 2)):
        super().__init__()
        assert isinstance(subsample_factor, tuple), "subsample_factor必须是tuple类型"

        if inter_channels is None:
            inter_channels = x_channels // 2
            if inter_channels == 0:
                inter_channels = 1

        # TODO: 这里的下采样系数subsample_factor有什么用，bias为什么为False
        self.x_conv = nn.Conv2d(x_channels, inter_channels, kernel_size=subsample_factor, stride=subsample_factor,
                                padding=0, bias=False)
        self.gate_conv = nn.Conv2d(gate_channels, inter_channels, kernel_size=1, stride=1, padding=0)
        self.psi = nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0)

        # 将最终乘以注意力系数的feature map再进行一次卷积
        # TODO: 为什么后面还要接一个BN
        self.end_transform = nn.Sequential(
            nn.Conv2d(x_channels, x_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(x_channels)
        )

        # TODO: 是否需要初始化参数

    def forward(self, x, gate):
        origin_x = x
        origin_x_size = x.size()

        x = self.x_conv(x)
        gate = self.gate_conv(gate)
        # 将卷积后的gate上采样为x的尺寸
        gate = F.upsample_bilinear(gate, size=x.size()[2:])

        # 将feature map相加后进行ReLU
        temp = F.relu(gate + x, inplace=True)
        temp = self.psi(temp)
        attention_coefficient = F.sigmoid(temp)

        # 将注意力系数上采样为原始x的尺寸，并扩展为对应的channel数
        attention_coefficient = F.upsample_bilinear(attention_coefficient, size=origin_x_size[2:])
        attention_coefficient = attention_coefficient.expand_as(origin_x)

        # 注意力系数乘原始的x
        output = attention_coefficient * origin_x
        output = self.end_transform(output)

        return output
