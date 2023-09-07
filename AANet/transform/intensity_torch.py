# -*- coding: utf-8 -*-
from __future__ import print_function, division

import json
import math
from scipy import ndimage
from skimage.util import random_noise
import random
import numpy as np
import torch
import torch.nn as nn
import numbers
from torch.nn import functional as F


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    def __init__(self, channels=1, kernel_size=5, sigma=0.5, dim=3):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels
        self.padding = [k // 2 for k in kernel_size]

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups, padding=self.padding)


class RandomBlur(nn.Module):
    """

    """

    def __init__(self, sigma=0.4, p=0.5, channel_apply=0):
        """

        """
        super().__init__()
        self.sigma = sigma
        self.channel_apply = channel_apply
        self.noise = GaussianSmoothing(channels=1, kernel_size=5, sigma=sigma)
        self.p = p

    def __call__(self, image):
        apply = torch.rand(image.shape[0], device=image.device) < self.p

        noise_image = image[:, self.channel_apply:self.channel_apply + 1]
        noise_image = self.noise(noise_image)

        t_image = image.clone()
        t_image[apply, self.channel_apply] = noise_image[apply, 0]

        return t_image


class RandomGamma(nn.Module):
    """

    """

    def __init__(self, gamma_range=1.4, p=0.5, channel_apply=0):
        """
        gamma range: gamme will be in [1/gamma_range, gamma_range]
        """
        super().__init__()

        self.gamma_range = gamma_range
        self.channel_apply = channel_apply
        self.p = p

    def __call__(self, image):
        apply = torch.rand(image.shape[0], 1, 1, 1) < self.p
        gamma_level = torch.rand(image.shape[0], 1, 1, 1) * (self.gamma_range - 1) + 1
        for i in range(gamma_level.shape[0]):
            if random.random() < 0.5:
                gamma_level[i] = 1. / gamma_level[i]

        gamma_level[apply == 0] = 1.
        noise_image = torch.pow(image[:, self.channel_apply], gamma_level.to(image.device))
        t_image = image.clone()
        t_image[:, self.channel_apply] = noise_image
        return t_image


class RandomNoise(nn.Module):
    """

    """

    def __init__(self, p=0.5, channel_apply=0, gamma_range=(1e-2, 2e-2)):
        """

        """
        super().__init__()

        self.p = p
        self.channel_apply = channel_apply
        self.gamma_range = gamma_range

    def __call__(self, image):
        apply = torch.rand(image.shape[0], 1, 1, 1, device=image.device) < self.p
        noise_level = torch.rand(image.shape[0], 1, 1, 1, device=image.device) * (
                self.gamma_range[1] - self.gamma_range[0]) + self.gamma_range[0]
        noise = torch.randn(image.shape, device=image.device)[:, 0] * noise_level * apply.float()

        noise_image = torch.clamp(image[:, self.channel_apply] + noise, min=0, max=1)
        t_image = image.clone()
        t_image[:, self.channel_apply] = noise_image
        return t_image
