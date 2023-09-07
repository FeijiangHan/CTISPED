import torch
from torch import nn
import math
import numpy as np
from torch.nn import functional as F


# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = False


def crack(integer):
    start = int(np.sqrt(integer))
    factor = integer / start
    while int(factor) != factor:
        start += 1
        factor = integer / start
    return int(factor), start


def activation(act='ReLU'):
    if act == 'ReLU':
        return nn.ReLU()
    elif act == 'LeakyReLU':
        return nn.LeakyReLU()
    elif act == 'ELU':
        return nn.ELU()
    elif act == 'PReLU':
        return nn.PReLU()
    else:
        return nn.Identity()


def norm_layer3d(norm_type, num_features):
    if norm_type == 'batchnorm':
        return nn.BatchNorm3d(num_features=num_features, momentum=0.05)
    elif norm_type == 'instancenorm':
        return nn.InstanceNorm3d(num_features=num_features)
    elif norm_type == 'groupnorm':
        return nn.GroupNorm(num_groups=num_features // 4, num_channels=num_features)
    else:
        return nn.Identity()


class StdConv3d(nn.Conv3d):
    """Conv2d with Weight Standardization. Used for BiT ResNet-V2 models.

    Paper: `Micro-Batch Training with Batch-Channel Normalization and Weight Standardization` -
        https://arxiv.org/abs/1903.10520v2
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1,
                 dilation=1, groups=1, bias=False, eps=1e-6):
        super().__init__(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.eps = eps

    def forward(self, x):
        weight = F.batch_norm(
            self.weight.view(1, self.out_channels, -1), None, None,
            training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        x = F.conv3d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x


class StdConvTranspose3d(nn.ConvTranspose3d):
    """Conv2d with Weight Standardization. Used for BiT ResNet-V2 models.

    Paper: `Micro-Batch Training with Batch-Channel Normalization and Weight Standardization` -
        https://arxiv.org/abs/1903.10520v2
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True,
                 dilation=1, eps=1e-6):
        super().__init__(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding,
            dilation=dilation, groups=groups, bias=bias)
        self.eps = eps

    def forward(self, x, output_size=None):
        output_padding = self._output_padding(input, output_size, self.stride, self.padding, self.kernel_size)
        weight = F.batch_norm(
            self.weight.permute(1, 0, 2, 3, 4).reshape(1, self.out_channels, -1), None, None,
            training=True, momentum=0., eps=self.eps).permute(1, 0, 2).reshape_as(self.weight)
        x = F.conv_transpose3d(
            x, weight, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)
        return x


class ClsHead(nn.Module):
    def __init__(self, in_channels, num_anchors=1, num_classes=1, feature_size=96, conv_num=2,
                 norm_type='groupnorm', act_type='ReLU'):
        super(ClsHead, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        cls_convs = []
        for i in range(conv_num):
            if i == 0:
                cls_convs.append(
                    ConvBlock(in_channels, feature_size, 3, norm_type=norm_type, act_type=act_type))
            else:
                cls_convs.append(
                    ConvBlock(feature_size, feature_size, 3, norm_type=norm_type, act_type=act_type))
        self.cls_convs = nn.Sequential(*cls_convs)
        self.cls_output = nn.Conv3d(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)
        self.cls_act = nn.Sigmoid()

    def forward(self, x):
        x_cls = self.cls_convs(x)
        x_cls = self.cls_act(self.cls_output(x_cls))

        # out is B x C x Z x Y x X, with C = n_classes * n_anchors
        x_cls = x_cls.permute(0, 2, 3, 4, 1)
        batch_size, zz, yy, xx, channels = x_cls.shape
        x_cls = x_cls.view(batch_size, zz, yy, xx, self.num_anchors, self.num_classes)

        return x_cls.contiguous().view(x_cls.shape[0], -1, self.num_classes)


class RegHead(nn.Module):
    def __init__(self, in_channels, num_anchors=1, num_classes=1, feature_size=96, conv_num=2,
                 norm_type='groupnorm', act_type='ReLU'):
        super(RegHead, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        reg_convs = []
        for i in range(conv_num):
            if i == 0:
                reg_convs.append(
                    ConvBlock(in_channels, feature_size, 3, norm_type=norm_type, act_type=act_type))
            else:
                reg_convs.append(
                    ConvBlock(feature_size, feature_size, 3, norm_type=norm_type, act_type=act_type))
        self.reg_convs = nn.Sequential(*reg_convs)
        self.reg_output = nn.Conv3d(feature_size, num_anchors * 6, kernel_size=3, padding=1)

    def forward(self, x):
        x_reg = self.reg_convs(x)
        x_reg = self.reg_output(x_reg)
        x_reg = x_reg.permute(0, 2, 3, 4, 1)

        return x_reg.contiguous().view(x.shape[0], -1, 6)


class SegHead(nn.Module):
    def __init__(self, in_channels, num_classes=1, feature_size=16,
                 norm_type='groupnorm', act_type='ReLU'):
        super(SegHead, self).__init__()

        self.num_classes = num_classes

        self.up_conv = UpsamplingDeconvBlock(in_channels, feature_size, stride=2,
                                             norm_type=norm_type, act_type=act_type)
        self.seg_output = nn.Conv3d(feature_size, num_classes, kernel_size=3, padding=1)
        self.seg_act = nn.Sigmoid()

    def forward(self, x):
        x_seg = self.up_conv(x)
        x_seg = self.seg_act(self.seg_output(x_seg))

        return x_seg


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, groups=1,
                 norm_type='groupnorm', act_type='ReLU'):
        super(ConvBlock, self).__init__()

        self.conv = StdConv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, groups=groups,
                              padding=kernel_size // 2 + dilation - 1, dilation=dilation, bias=False)
        self.norm = norm_layer3d(norm_type, out_channels)
        self.act = activation(act_type)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, norm_type='groupnorm', act_type='ReLU'):
        super(BasicBlock, self).__init__()

        self.conv1 = ConvBlock(in_channels=in_channels, out_channels=out_channels, stride=stride,
                               act_type=act_type, norm_type=norm_type)

        self.conv2 = ConvBlock(in_channels=out_channels, out_channels=out_channels, stride=1,
                               act_type='none', norm_type=norm_type)

        if in_channels == out_channels and stride == 1:
            self.res = nn.Identity()
        elif in_channels != out_channels and stride == 1:
            self.res = ConvBlock(in_channels, out_channels, kernel_size=1, act_type='none', norm_type=norm_type)
        elif in_channels != out_channels and stride > 1:
            self.res = nn.Sequential(
                nn.AvgPool3d(kernel_size=2, stride=2),
                ConvBlock(in_channels, out_channels, kernel_size=1, act_type='none', norm_type=norm_type))

        self.act = activation(act_type)

    def forward(self, x):
        identity = self.res(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x += identity
        x = self.act(x)

        return x


class LayerBasic(nn.Module):
    def __init__(self, n_stages, in_channels, out_channels, stride=1, norm_type='groupnorm', act_type='ReLU'):
        super(LayerBasic, self).__init__()
        self.n_stages = n_stages
        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = in_channels
                stride = stride
            else:
                input_channel = out_channels
                stride = 1

            ops.append(BasicBlock(input_channel, out_channels, stride=stride, norm_type=norm_type, act_type=act_type))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class DownsamplingConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, norm_type='groupnorm', act_type='ReLU'):
        super(DownsamplingConvBlock, self).__init__()

        self.conv = StdConv3d(in_channels, out_channels, kernel_size=2, padding=0, stride=stride, bias=False)
        self.norm = norm_layer3d(norm_type, out_channels)
        self.act = activation(act_type)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class DownsamplingBlock(nn.Module):
    def __init__(self, in_channels=None, out_channels=None, stride=2, pool_type='max',
                 norm_type='groupnorm', act_type='ReLU'):
        super(DownsamplingBlock, self).__init__()

        if pool_type == 'avg':
            self.down = nn.AvgPool3d(kernel_size=stride, stride=stride)
        else:
            self.down = nn.MaxPool3d(kernel_size=stride, stride=stride)
        if (in_channels is not None) and (out_channels is not None):
            self.conv = ConvBlock(in_channels, out_channels, 1, norm_type=norm_type, act_type=act_type)

    def forward(self, x):
        x = self.down(x)
        if hasattr(self, 'conv'):
            x = self.conv(x)
        return x


class UpsamplingDeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, norm_type='groupnorm', act_type='ReLU'):
        super(UpsamplingDeconvBlock, self).__init__()

        self.conv = StdConvTranspose3d(in_channels, out_channels, kernel_size=stride, padding=0, stride=stride,
                                       bias=False)
        self.norm = norm_layer3d(norm_type, out_channels)
        self.act = activation(act_type)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class UpsamplingBlock(nn.Module):
    def __init__(self, in_channels=None, out_channels=None, stride=2, mode='nearest', norm_type='groupnorm',
                 act_type='ReLU'):
        super(UpsamplingBlock, self).__init__()

        self.up = nn.Upsample(scale_factor=stride, mode=mode)
        if (in_channels is not None) and (out_channels is not None):
            self.conv = ConvBlock(in_channels, out_channels, 1, norm_type=norm_type, act_type=act_type)

    def forward(self, x):
        if hasattr(self, 'conv'):
            x = self.conv(x)
        x = self.up(x)
        return x


class ASPP(nn.Module):
    def __init__(self, channels, out_channels, ratio=4,
                 dilations=[1, 2, 3, 4],
                 norm_type='batchnorm', act_type='ReLU'):
        super(ASPP, self).__init__()
        # assert dilations[0] == 1, 'The first item in dilations should be `1`'
        inner_channels = channels // ratio
        cat_channels = inner_channels * 5
        self.aspp0 = ConvBlock(channels, inner_channels, kernel_size=1,
                               dilation=dilations[0], norm_type=norm_type, act_type=act_type)
        self.aspp1 = ConvBlock(channels, inner_channels, kernel_size=3,
                               dilation=dilations[1], norm_type=norm_type, act_type=act_type)
        self.aspp2 = ConvBlock(channels, inner_channels, kernel_size=3,
                               dilation=dilations[2], norm_type=norm_type, act_type=act_type)
        self.aspp3 = ConvBlock(channels, inner_channels, kernel_size=3,
                               dilation=dilations[3], norm_type=norm_type, act_type=act_type)
        self.avg_conv = nn.Sequential(nn.AdaptiveAvgPool3d(1),
                                      ConvBlock(channels, inner_channels, kernel_size=1,
                                                dilation=1, norm_type=norm_type, act_type=act_type))
        self.transition = ConvBlock(cat_channels, out_channels, kernel_size=1,
                                    dilation=dilations[0], norm_type=norm_type, act_type=act_type)

    def forward(self, input):
        aspp0 = self.aspp0(input)
        aspp1 = self.aspp1(input)
        aspp2 = self.aspp2(input)
        aspp3 = self.aspp3(input)
        avg = self.avg_conv(input)
        avg = F.interpolate(avg, aspp2.size()[2:], mode='nearest')
        out = torch.cat((aspp0, aspp1, aspp2, aspp3, avg), dim=1)
        out = self.transition(out)
        return out


class ASPP2(nn.Module):
    def __init__(self, channels, out_channels, ratio=4,
                 dilations=[1, 2, 3, 4, 5],
                 norm_type='groupnorm', act_type='ReLU'):
        super(ASPP2, self).__init__()
        # assert dilations[0] == 1, 'The first item in dilations should be `1`'
        inner_channels = channels // ratio
        cat_channels = inner_channels * 5
        self.aspp0 = ConvBlock(channels, inner_channels, kernel_size=1,
                               dilation=dilations[0], norm_type=norm_type, act_type=act_type)
        self.aspp1 = ConvBlock(channels, inner_channels, kernel_size=3,
                               dilation=dilations[1], norm_type=norm_type, act_type=act_type)
        self.aspp2 = ConvBlock(channels, inner_channels, kernel_size=3,
                               dilation=dilations[2], norm_type=norm_type, act_type=act_type)
        self.aspp3 = ConvBlock(channels, inner_channels, kernel_size=3,
                               dilation=dilations[3], norm_type=norm_type, act_type=act_type)
        self.aspp4 = ConvBlock(channels, inner_channels, kernel_size=3,
                               dilation=dilations[4], norm_type=norm_type, act_type=act_type)
        self.transition = ConvBlock(cat_channels, out_channels, kernel_size=1,
                                    dilation=dilations[0], norm_type=norm_type, act_type=act_type)

    def forward(self, input):
        aspp0 = self.aspp0(input)
        aspp1 = self.aspp1(input)
        aspp2 = self.aspp2(input)
        aspp3 = self.aspp3(input)
        aspp4 = self.aspp4(input)

        out = torch.cat((aspp0, aspp1, aspp2, aspp3, aspp4), dim=1)
        out = self.transition(out)
        return out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y
