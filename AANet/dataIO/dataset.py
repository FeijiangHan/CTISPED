# -*- coding: utf-8 -*-
from __future__ import print_function, division

import os
import torch
from torch import randperm
import pandas as pd
import numpy as np
import random
import SimpleITK as sitk
from torch.utils.data import Dataset
from scipy import ndimage
import time
import re
import sys
from . import utils


class DatasetNumpy(Dataset):
    """Dataset for loading images. It generates 4D tensors with
    dimention order [C, D, H, W] for 3D images, and 3D tensors
    with dimention order [C, H, W] for 2D images"""

    def __init__(self, image_root, label_root, file_name, transform_post=None, crop_fn=None, replicate=1, ):

        self.image_file_list = sorted([os.path.join(image_root, x) for x in file_name if x.endswith(".npy")])

        self.label_file_list = sorted([os.path.join(label_root, x) for x in file_name if x.endswith(".npy")])

        self.transform_post = transform_post
        self.crop_fn = crop_fn
        self.replicate = replicate

    def __len__(self):
        return len(self.image_file_list) * self.replicate

    def __getitem__(self, idx):
        idx = idx // self.replicate
        image_path = self.image_file_list[idx]
        image = np.load(image_path, allow_pickle=True)

        label_path = self.label_file_list[idx]
        label = np.load(label_path, allow_pickle=True)

        assert image.shape == label.shape, self.image_file_list[idx]

        data = {}
        data['image'] = image.astype('float32')
        data['label'] = label.astype('uint8')

        samples = self.crop_fn(data)
        transformed_samples = []

        for i in range(len(samples)):
            sample = samples[i]
            if self.transform_post:
                sample = self.transform_post(sample)
            transformed_samples.append(sample)

        return transformed_samples


class DatasetITK(Dataset):
    """Dataset for loading images. It generates 4D tensors with
    dimention order [C, D, H, W] for 3D images, and 3D tensors
    with dimention order [C, H, W] for 2D images"""

    def __init__(self, image_root, label_root, file_name, transform_post=None, crop_fn=None, replicate=1, ):

        self.image_file_list = sorted([os.path.join(image_root, x) for x in file_name if x.endswith(".nii.gz")])

        self.label_file_list = sorted([os.path.join(label_root, x) for x in file_name if x.endswith(".nii.gz")])

        self.transform_post = transform_post
        self.crop_fn = crop_fn
        self.replicate = replicate

    def __len__(self):
        return len(self.image_file_list) * self.replicate

    def __getitem__(self, idx):
        idx = idx // self.replicate
        image_path = self.image_file_list[idx]
        image_itk = sitk.ReadImage(image_path)

        label_path = self.label_file_list[idx]
        label_itk = sitk.ReadImage(label_path)

        assert image_itk.GetSize() == label_itk.GetSize(), self.image_file_list[idx]

        data = {}
        data['image_itk'] = image_itk
        data['label_itk'] = label_itk

        samples = self.crop_fn(data)
        transformed_samples = []

        for i in range(len(samples)):
            sample = samples[i]
            if self.transform_post:
                sample = self.transform_post(sample)
            transformed_samples.append(sample)

        return transformed_samples


class DatasetITKV(Dataset):
    """Dataset for loading images. It generates 4D tensors with
    dimention order [C, D, H, W] for 3D images, and 3D tensors
    with dimention order [C, H, W] for 2D images"""

    def __init__(self, image_root, label_root, vessel_root, file_name, transform_post=None, crop_fn=None, replicate=1):

        self.image_file_list = sorted([os.path.join(image_root, x) for x in file_name if x.endswith(".nii.gz")])
        self.label_file_list = sorted([os.path.join(label_root, x) for x in file_name if x.endswith(".nii.gz")])
        self.vessel_file_list = sorted([os.path.join(vessel_root, x) for x in file_name if x.endswith(".nii.gz")])

        self.transform_post = transform_post
        self.crop_fn = crop_fn
        self.replicate = replicate

    def __len__(self):
        return len(self.image_file_list)

    def __getitem__(self, idx):
        # print(idx)
        image_path = self.image_file_list[idx]
        image_itk = sitk.ReadImage(image_path)

        label_path = self.label_file_list[idx]
        label_itk = sitk.ReadImage(label_path)

        vessel_path = self.vessel_file_list[idx]
        vessel_itk = sitk.ReadImage(vessel_path)

        assert image_itk.GetSize() == label_itk.GetSize(), self.image_file_list[idx]
        assert image_itk.GetSize() == vessel_itk.GetSize(), self.image_file_list[idx]

        data = {}
        data['image_itk'] = image_itk
        data['label_itk'] = label_itk
        data['vessel_itk'] = vessel_itk

        samples = self.crop_fn(data)
        transformed_samples = []

        for i in range(len(samples)):
            sample = samples[i]
            if self.transform_post:
                sample = self.transform_post(sample)
            transformed_samples.append(sample)

        return transformed_samples


def collate_fn_dict(batches):
    batch = []
    [batch.extend(b) for b in batches]
    image = [s['image'] for s in batch]
    image = np.stack(image)
    label = [s['label'] for s in batch]
    label = np.stack(label)
    annots = [s['annot'] for s in batch]
    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:
        annot_padded = np.ones((len(annots), max_num_annots, 7), dtype='float32') * -1
        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = np.ones((len(annots), 1, 7), dtype='float32') * -1

    return {'image': torch.tensor(image), 'label': torch.tensor(label), 'annot': torch.tensor(annot_padded)}


def collate_fn_dictv(batches):
    batch = []
    [batch.extend(b) for b in batches]
    image = [s['image'] for s in batch]
    image = np.stack(image)
    label = [s['label'] for s in batch]
    label = np.stack(label)
    mask = [s['mask'] for s in batch]
    mask = np.stack(mask)

    return {'image': torch.tensor(image), 'label': torch.tensor(label), 'mask': torch.tensor(mask)}


def collate_fn_dicts(batches):
    batch = []
    [batch.extend(b) for b in batches]
    sample = {}
    for key in batch[0]:
        sample[key] = torch.tensor(np.stack([s[key] for s in batch]))

    return sample
