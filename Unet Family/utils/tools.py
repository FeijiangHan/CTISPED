#!/usr/bin/python3
# -*- coding: utf-8 -*
from torch import nn
import json
import os
import torch
import shutil
import numpy as np
from PIL import ImageDraw, Image
import random


def generate_noisy_indexes(seed, dataset_size, noisy_rate):
    """生成需要加噪的图像的index"""

    num = int(dataset_size * noisy_rate)
    random.seed(seed)
    indexes = random.sample(range(0, dataset_size), num)
    return indexes


def add_noisy_triangle_label(mask):
    """在mask上添加噪声标注"""

    while True:
        noisy_image = Image.fromarray(np.zeros(mask.size, dtype=np.uint8))
        # 生成随机的三角形的顶点坐标
        noisy_points = np.random.randint(0, 256, (3, 2))
        # 三角形的跨度不能太大
        if (noisy_points.max(axis=0) - noisy_points.min(axis=0)).max() > 75:
            continue

        noisy_points = noisy_points.flatten()
        noisy_points = tuple(noisy_points)
        ImageDraw.Draw(noisy_image).polygon(noisy_points, fill=255)

        # 噪声标签的面积不能太大
        noisy_array = np.asarray(noisy_image) / 255
        if noisy_array.sum() <= 1500 and noisy_array.sum() >= 1000:
            # 在原图上添加噪声
            ImageDraw.Draw(mask).polygon(noisy_points, fill=255)
            break

    return mask


def create_directory(dir_path):
    """创建目录，若存在，则删除后再创建"""

    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        print('Remove {}'.format(dir_path))
    os.mkdir(dir_path)
    print('Create {}'.format(dir_path))


def process_binary_mask_tensor(mask):
    """对生成的Tensor类型的mask进行处理"""

    assert isinstance(mask, torch.Tensor), '输入的不是Tensor类型'

    # ToTensor()会自动给数据添加一个channel维度，因为是类别标签，不需要这个维度
    if len(mask.shape) == 3:
        mask = torch.squeeze(mask, dim=0)

    mask = mask.to(torch.uint8)
    mask[mask != 0] = 1

    return mask


def process_multiple_mask_tensor(mask, color_map):
    """对生成的Tensor类型的mask进行处理"""

    assert isinstance(mask, torch.Tensor), '输入的不是Tensor类型'

    # ToTensor()会自动给数据添加一个channel维度，因为是类别标签，不需要这个维度
    if len(mask.shape) == 3:
        mask = torch.squeeze(mask, dim=0)

    # 如果mask被Resize，修改其中生成的噪声标签，并将数据类型转换为uint8，将其中的255替换为1
    if torch.max(mask).item() <= 1:
        # ToTensor()会将数据进行max-min归一化，这里进行还原
        mask = (mask * 255).to(torch.uint8)

        # 记录噪声标记的坐标
        noise_indexes = torch.ones_like(mask, dtype=torch.bool)

        # 去除噪声标记
        for color_value, class_id in color_map.items():
            noise_indexes = noise_indexes & (mask != color_value)
        mask[noise_indexes] = 0

        # 将对应的类别的颜色值替换为其类别id
        for color_value, class_id in color_map.items():
            mask[mask == color_value] = class_id

        # 因为Resize使用线性插值的缘故，会产生一些少量的现有类别的标记，但是其数量很少，将其过滤掉
        for class_id in color_map.values():
            if torch.sum(mask == class_id).item() <= 10:
                mask[mask == class_id] = 0

    return mask


def is_augmented(path):
    """从路径判断是否是被增广的数据"""

    augmented_types = ['VerticalFlip', 'HorizontalFlip', 'Transpose', 'RandomRotate90']
    for t in augmented_types:
        if path.__contains__(t):
            return True
    return False


def remove_augmented_item(old_items):
    """删除被数据增强的数据"""

    new_items = []
    for item in old_items:
        if not is_augmented(item['image_path']):
            new_items.append(item)
    return new_items


def make_dataset(mode, base_path, is_contain_augmented_data):
    """从指定目录下读取用于training、validation的json文件"""

    assert mode in ['train', 'val']

    path = os.path.join(base_path, mode)
    image_path = os.path.join(path, "image")
    mask_path = os.path.join(path, "mask")

    image_list = os.listdir(image_path)
    mask_list = os.listdir(mask_path)
    return image_list, mask_list

    # json_path = os.path.join(base_path, '{}.json'.format(mode))
    #
    # with open(json_path, 'r') as f:
    #     items = json.load(f)
    #
    # if not is_contain_augmented_data and mode == 'train':
    #     items = remove_augmented_item(items)
    #
    # return items


def initialize_weights(model):
    """初始化神经网络中的参数"""

    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.BatchNorm2d):
            module.weight.data.fill_(1)
            module.bias.data.zero_()
