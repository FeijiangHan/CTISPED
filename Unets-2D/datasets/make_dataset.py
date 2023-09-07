# -*- coding: utf-8 -*-
import os


def make_dataset(mode, base_path, is_contain_augmented_data):
    """从指定目录下读取用于training、validation的json文件"""

    assert mode in ['train', 'val']

    path = os.path.join(base_path, mode)
    image_path = os.path.join(path, "image")
    mask_path = os.path.join(path, "mask")

    image_list = os.listdir(image_path)
    mask_list = os.listdir(mask_path)
    return image_list, mask_list