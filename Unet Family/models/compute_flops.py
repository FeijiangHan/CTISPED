# -*- coding: utf-8 -*-


import torchvision.models as models
import torch
from ptflops import get_model_complexity_info
from models import unet, nested_unet, loss_function, xnet, nfn_plus, lovasz_losses, unet_tiny, U2net


with torch.cuda.device(0):
    net = U2net.U2NET(out_ch=2, in_ch=3)
    # net = unet.UNet(num_classes=2, in_channels=3, is_attention=True,
    #                 is_recurrent_residual=False)
# net = models.densenet161()
    macs, params = get_model_complexity_info(net, (3, 128, 128), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

"""
U2NETP: Number of parameters:           1.13 M 
U2NET:  Number of parameters:           44.02 M 
"""