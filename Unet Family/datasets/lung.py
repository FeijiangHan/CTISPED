#!/usr/bin/python3
# -*- coding: utf-8 -*
from torch.utils.data import Dataset
import torch
from PIL import Image
from utils.tools import process_binary_mask_tensor
# from datasets.make_dataset import make_dataset
import os
import numpy as np


class Lung(Dataset):

    CHANNELS_NUM = 3
    NUM_CLASSES = 2

    MEAN = [0.70791537, 0.59156666, 0.54687498]
    STD = [0.15324752, 0.16178547, 0.17681521]

    def __init__(self, mode, transform=None, target_transform=None, BASE_PATH=""):
        print(mode)
        self.items_image, self.items_mask = make_dataset(mode, BASE_PATH)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.items_image)

    def __str__(self):
        return 'Stroke'

    def __getitem__(self, index):
        # image_path = self.items[index]['image_path']
        # mask_path = self.items[index]['mask_path']
        image_path = self.items_image[index]
        mask_path = self.items_mask[index]

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)

        mask = process_binary_mask_tensor(mask)
        # print(torch.unique(mask))

        return image, mask


def make_dataset(mode, base_path):
    print(mode)

    # assert mode in ['train', 'val']
    #
    # path = os.path.join(base_path, mode)
    image_path = os.path.join(base_path, "image")
    mask_path = os.path.join(base_path, "mask")
    # print(image_path)
    image_list = []
    for file in os.listdir(image_path):
        image_list.append(os.path.join(image_path, file))

    mask_list = []
    for file in os.listdir(mask_path):
        mask_list.append(os.path.join(mask_path, file))

    # print(image_list)
    return image_list, mask_list

