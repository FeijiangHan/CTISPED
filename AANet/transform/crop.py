# -*- coding: utf-8 -*-
from __future__ import print_function, division

import json
import random
from .abstract_transform import AbstractTransform
import numpy as np
from scipy import ndimage


class RandomCrop(object):
    """Randomly crop the input image (shape [C, D, H, W] or [C, H, W])
    """

    def __init__(self, output_size):
        """

        """
        self.output_size = output_size

        assert isinstance(self.output_size, (list, tuple))

    def __call__(self, sample):
        image = sample['image']
        input_shape = image.shape
        input_dim = len(input_shape) - 1

        crop_margin = [input_shape[i + 1] - self.output_size[i] \
                       for i in range(input_dim)]

        bb_min = [0] * (input_dim + 1)
        bb_max = image.shape
        bb_min, bb_max = bb_min[1:], bb_max[1:]
        crop_min = [random.randint(bb_min[i], bb_max[i]) - int(self.output_size[i] / 2) \
                    for i in range(input_dim)]
        crop_min = [max(0, item) for item in crop_min]
        crop_min = [min(crop_min[i], input_shape[i + 1] - self.output_size[i]) for i in range(input_dim)]

        crop_max = [crop_min[i] + self.output_size[i] for i in range(input_dim)]

        crop_min = [0] + crop_min
        crop_max = list(input_shape[0:1]) + crop_max

        image_t = crop_ND_volume_with_bounding_box(image, crop_min, crop_max)
        sample['image'] = image_t

        if 'coord' in sample:
            sample['coord'] = sample['coord'] - crop_min[1:]

        if 'label' in sample:
            label = sample['label']
            label = crop_ND_volume_with_bounding_box(label, crop_min, [label.shape[0]] + crop_max[1:])
            sample['label'] = label
        if 'mask' in sample:
            mask = sample['mask']
            mask = crop_ND_volume_with_bounding_box(mask, crop_min, [mask.shape[0]] + crop_max[1:])
            sample['mask'] = mask

        return sample


def crop_ND_volume_with_bounding_box(volume, min_idx, max_idx):
    """
    crop/extract a subregion form an nd image.
    """
    dim = len(volume.shape)
    assert (dim >= 2 and dim <= 5)
    assert (max_idx[0] - min_idx[0] <= volume.shape[0])
    if (dim == 2):
        output = volume[np.ix_(range(min_idx[0], max_idx[0]),
                               range(min_idx[1], max_idx[1]))]
    elif (dim == 3):
        output = volume[np.ix_(range(min_idx[0], max_idx[0]),
                               range(min_idx[1], max_idx[1]),
                               range(min_idx[2], max_idx[2]))]
    elif (dim == 4):
        output = volume[np.ix_(range(min_idx[0], max_idx[0]),
                               range(min_idx[1], max_idx[1]),
                               range(min_idx[2], max_idx[2]),
                               range(min_idx[3], max_idx[3]))]
    elif (dim == 5):
        output = volume[np.ix_(range(min_idx[0], max_idx[0]),
                               range(min_idx[1], max_idx[1]),
                               range(min_idx[2], max_idx[2]),
                               range(min_idx[3], max_idx[3]),
                               range(min_idx[4], max_idx[4]))]
    else:
        raise ValueError("the dimension number shoud be 2 to 5")
    return output
