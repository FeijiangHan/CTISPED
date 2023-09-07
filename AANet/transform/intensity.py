# -*- coding: utf-8 -*-
from __future__ import print_function, division

import json
import math
from .abstract_transform import AbstractTransform
from scipy import ndimage
from skimage.util import random_noise
import random
import numpy as np


class RandomBlur(AbstractTransform):
    """

    """

    def __init__(self, sigma_range=(0.4, 0, 8), p=0.5, channel_apply=0):
        """

        """
        self.sigma_range = sigma_range
        self.channel_apply = channel_apply
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            image = sample['image']
            sigma = np.random.uniform(self.sigma_range[0], self.sigma_range[1])
            image_t = ndimage.gaussian_filter(image[self.channel_apply], sigma)
            image[self.channel_apply] = image_t
            sample['image'] = image

        return sample


class RandomGamma(AbstractTransform):
    """

    """

    def __init__(self, gamma_range=2, p=0.5, channel_apply=0):
        """
        gamma range: gamme will be in [1/gamma_range, gamma_range]
        """
        self.gamma_range = gamma_range
        self.channel_apply = channel_apply
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            image = sample['image']
            gamma = np.random.uniform(1, self.gamma_range)
            if random.random() < 0.5:
                gamma = 1. / gamma
            image_t = np.power(image[self.channel_apply], gamma)
            image[self.channel_apply] = image_t
            sample['image'] = image

        return sample


class RandomNoise(AbstractTransform):
    """

    """

    def __init__(self, p=0.5, channel_apply=0, gamma_range=(1e-4, 5e-4)):
        """

        """
        self.p = p
        self.channel_apply = channel_apply
        self.gamma_range = gamma_range

    def __call__(self, sample):
        if random.random() < self.p:
            image = sample['image']
            gamma = np.random.uniform(self.gamma_range[0], self.gamma_range[1])
            image_t = random_noise(image[self.channel_apply], var=gamma)
            image[self.channel_apply] = image_t
            sample['image'] = image

        return sample
