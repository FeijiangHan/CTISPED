#!/usr/bin/python3
# -*- coding: utf-8 -*-

# UNet implementation using ResNet34 as the encoder
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models


class _DecoderBlock(nn.Module):
    """
    Decoder block for UNet - two convolutions 
    followed by upsampling, except for last block
    """

    def __init__(self, in_channels, middle_channels, out_channels=None):
        super().__init__()

        # Two convolutions followed by upsampling 
        layers = [
            nn.Conv2d(in_channels, middle_channels, kernel_size=3),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, kernel_size=3),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True)
        ]

        if out_channels:
            # Final upsampling 
            layers.append(nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2))

        self.decode = nn.Sequential(*layers)

    def forward(self, input):
        return self.decode(input)


class UNet_ResNet34(nn.Module):
    """
    UNet implementation using pretrained ResNet34 as encoder.
    ResNet34 has 4 stages which aligns well with UNet downsampling. 
    """

    def __init__(self, num_classes):
        super().__init__()

        # Load pretrained ResNet34
        resnet34 = models.resnet34(pretrained=True)
        
        # Encoder stages
        self.enc1 = nn.Sequential(
            resnet34.conv1,
            resnet34.bn1,
            resnet34.relu,
            resnet34.maxpool,
            resnet34.layer1
        )
        self.enc2 = resnet34.layer2
        self.enc3 = resnet34.layer3
        self.enc4 = resnet34.layer4

        # Center block
        self.center = _DecoderBlock(512, 1024, 512)

        # Decoder blocks
        self.dec4 = _DecoderBlock(1024, 512, 256) 
        self.dec3 = _DecoderBlock(512, 256, 128)
        self.dec2 = _DecoderBlock(256, 128, 64)
        self.dec1 = _DecoderBlock(128, 64)

        # Final convolution
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, input):
        
        # Encoder
        enc1 = self.enc1(input) 
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)  
        enc4 = self.enc4(enc3)
        
        # Center block
        center = self.center(enc4)
        
        # Decoder with skip connections
        dec4 = self.dec4(self.feature_fusion(enc4, center))
        dec3 = self.dec3(self.feature_fusion(enc3, dec4))
        dec2 = self.dec2(self.feature_fusion(enc2, dec3))
        dec1 = self.dec1(self.feature_fusion(enc1, dec2))
        
        # Final layer
        final = self.final(dec1)
        
        # Upsample to input size
        return F.upsample_bilinear(final, input.shape[2:]) 

    def feature_fusion(self, t1, t2):
        """
        Fuse features from encoder and decoder by matching 
        dimensions using bilinear interpolation.
        """

        # Assume square inputs
        if t1.shape[-1] >= t2.shape[-1]:
            large = t1
            small = t2
        else:
            large = t2 
            small = t1

        return torch.cat([F.upsample_bilinear(large, small.shape[2:]), small], dim=1)