import torch
from torch import nn
import math
import numpy as np
from .StdBlocks import *
import model_utils


# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = False


class Net(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, en_blocks=[2, 3, 3, 3], de_blocks=[2, 2, 2],
                 n_filters=[64, 96, 128, 128],
                 stem_filters=32, aspp_filters=32, norm_type='groupnorm', head_norm='groupnorm', act_type='ReLU'):
        super(Net, self).__init__()

        self.in_conv = ConvBlock(n_channels, stem_filters, stride=1, norm_type=norm_type, act_type=act_type)
        self.in_dw = ConvBlock(stem_filters, n_filters[0], stride=2, norm_type=norm_type, act_type=act_type)

        self.block1 = LayerBasic(en_blocks[0], n_filters[0], n_filters[0], norm_type=norm_type, act_type=act_type)
        self.block1_dw = DownsamplingConvBlock(n_filters[0], n_filters[1], norm_type=norm_type, act_type=act_type)

        self.block2 = LayerBasic(en_blocks[1], n_filters[1], n_filters[1], norm_type=norm_type, act_type=act_type)
        self.block2_dw = DownsamplingConvBlock(n_filters[1], n_filters[2], norm_type=norm_type, act_type=act_type)

        self.block3 = LayerBasic(en_blocks[2], n_filters[2], n_filters[2], norm_type=norm_type, act_type=act_type)
        self.block3_dw = DownsamplingConvBlock(n_filters[2], n_filters[3], norm_type=norm_type, act_type=act_type)

        self.block4 = LayerBasic(en_blocks[3], n_filters[3], n_filters[3], norm_type=norm_type, act_type=act_type)
        self.block44_aspp = UpsamplingBlock(n_filters[3], aspp_filters, stride=8, norm_type=norm_type,
                                            act_type=act_type)

        self.block33_up = UpsamplingDeconvBlock(n_filters[3], n_filters[2], norm_type=norm_type, act_type=act_type)
        self.block33_res = LayerBasic(1, n_filters[2], n_filters[2], norm_type=norm_type, act_type=act_type)
        self.block33 = LayerBasic(de_blocks[2], n_filters[2] * 2, n_filters[2], norm_type=norm_type, act_type=act_type)
        self.block33_aspp = UpsamplingBlock(n_filters[2], aspp_filters, stride=4, norm_type=norm_type,
                                            act_type=act_type)

        self.block22_up = UpsamplingDeconvBlock(n_filters[2], n_filters[1], norm_type=norm_type, act_type=act_type)
        self.block22_res = LayerBasic(1, n_filters[1], n_filters[1], norm_type=norm_type, act_type=act_type)
        self.block22 = LayerBasic(de_blocks[1], n_filters[1] * 2, n_filters[1], norm_type=norm_type, act_type=act_type)
        self.block22_aspp = UpsamplingBlock(n_filters[1], aspp_filters, stride=2, norm_type=norm_type,
                                            act_type=act_type)

        self.block11_up = UpsamplingDeconvBlock(n_filters[1], aspp_filters, norm_type=norm_type, act_type=act_type)
        self.block11_res = LayerBasic(1, n_filters[0], aspp_filters, norm_type=norm_type, act_type=act_type)
        self.block11 = LayerBasic(de_blocks[0], aspp_filters * 2, aspp_filters, norm_type=norm_type, act_type=act_type)
        self.aspp = ASPP2(aspp_filters * 4, aspp_filters, ratio=4, dilations=[1, 3, 5, 7, 9], norm_type=norm_type)

        self.seghead = SegHead(in_channels=aspp_filters, feature_size=16,
                               num_classes=n_classes, norm_type=head_norm, act_type=act_type)
        self.segheadv = SegHead(in_channels=aspp_filters, feature_size=16,
                                num_classes=1, norm_type=head_norm, act_type=act_type)
        self.init_weight()

    def forward(self, input):
        ct = input[:, :1]
        "input encode"
        x = self.in_conv(ct)
        x = self.in_dw(x)

        x1 = self.block1(x)
        x = self.block1_dw(x1)

        x2 = self.block2(x)
        x = self.block2_dw(x2)

        x3 = self.block3(x)
        x = self.block3_dw(x3)

        x = self.block4(x)
        x4 = self.block44_aspp(x)

        "decode"
        x = self.block33_up(x)
        x3 = self.block33_res(x3)
        x = torch.cat([x, x3], dim=1)
        x = self.block33(x)
        x3 = self.block33_aspp(x)

        x = self.block22_up(x)
        x2 = self.block22_res(x2)
        x = torch.cat([x, x2], dim=1)
        x = self.block22(x)
        x2 = self.block22_aspp(x)

        x = self.block11_up(x)
        x1 = self.block11_res(x1)
        x = torch.cat([x, x1], dim=1)
        x = self.block11(x)

        x = self.aspp(torch.cat([x, x2, x3, x4], dim=1))
        segv = self.segheadv(x)
        segv_down = F.max_pool3d(segv, kernel_size=2, stride=2).repeat(1, x.shape[1], 1, 1, 1)

        x = x * segv_down
        seg = self.seghead(x)

        return {'segs': seg, 'segvs': segv}

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm, nn.InstanceNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.init_head()

    def init_head(self):
        prior = 0.01
        nn.init.constant_(self.seghead.seg_output.weight, 0)
        nn.init.constant_(self.seghead.seg_output.bias, -math.log((1.0 - prior) / prior))
        nn.init.constant_(self.segheadv.seg_output.weight, 0)
        nn.init.constant_(self.segheadv.seg_output.bias, -math.log((1.0 - prior) / prior))
