#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import random

import numpy as np
from scipy import io
import cv2
import matplotlib.pyplot as plt
import h5py
import glob
import pydicom


def to255(img):     # 归一化到0-255代码
    min_val = img.min()
    max_val = img.max()
    img = (img - min_val) / (max_val - min_val + 1e-5)  # 图像归一化
    img = img * 255  # *255
    return img

src_path = r'C:\data\TianChiCTSeg\GroundTruth'  # mat文件的上级文件夹路径，英文路径
save_path = r'C:\data\TianChiCTSeg\data' # 保存png路径

train_image_path = save_path + '\\' + 'train\\' + 'image'
test_image_path = save_path + '\\' + 'val\\' + 'image'
train_mask_path = save_path + '\\' + 'train\\' + 'mask'
test_mask_path = save_path + '\\' + 'val\\' + 'mask'

os.makedirs(train_image_path, exist_ok=True)
os.makedirs(test_image_path, exist_ok=True)
os.makedirs(train_mask_path, exist_ok=True)
os.makedirs(test_mask_path, exist_ok=True)

names = glob.glob(src_path + '\\' + '*.mat')
random.shuffle(names)
order = 0
for name in names:
    # feature=h5py.File(name)               #读取mat文件
    feature = io.loadmat(name)  # (512, 512, 183), ['__header__', '__version__', '__globals__', 'Mask']
    # print(feature.keys())
    # 获取病人的名字
    this_pat_name = name.split('\\')[-1].split('.')[0]
    # 寻找病人对应的image的路径
    pat_image_root_path = r'C:\data\TianChiCTSeg\CT_scans 01' + '\\' + this_pat_name
    # 首先获取mask
    this_3D_mask = feature['Mask']
    # 下面进行到255的转换
    this_3D_mask[np.where(this_3D_mask == 1)] = 255
    # 下面进行slice的切分
    x, y, z = this_3D_mask.shape
    if order % 5 == 0:
        # 验证集
        for slice_index in range(z):
            this_2D_mask = this_3D_mask[:, :, slice_index]
            # 进行保存
            this_save_name = this_pat_name + '_' + 'D' + str(slice_index+1).zfill(4) + '.png'
            cv2.imwrite(test_mask_path + '\\' + this_save_name, this_2D_mask)
            # 读取原始dcm图像
            ds = pydicom.dcmread(pat_image_root_path + '\\' + 'D' + str(slice_index+1).zfill(4) + '.dcm')
            data = ds.pixel_array   # 0-4095， 肺部窗宽 1 300 Hu~1 700 Hu,窗位-600 Hu~-800 Hu,
            ct_image = to255(data)
            cv2.imwrite(test_image_path + '\\' + this_save_name, ct_image)
    else:
        # 训练集
        for slice_index in range(z):
            this_2D_mask = this_3D_mask[:, :, slice_index]
            # 进行保存
            this_save_name = this_pat_name + '_' + 'D' + str(slice_index+1).zfill(4) + '.png'
            cv2.imwrite(train_mask_path + '\\' + this_save_name, this_2D_mask)
            # 读取原始dcm图像
            ds = pydicom.dcmread(pat_image_root_path + '\\' + 'D' + str(slice_index+1).zfill(4) + '.dcm')
            data = ds.pixel_array   # 0-4095， 肺部窗宽 1 300 Hu~1 700 Hu,窗位-600 Hu~-800 Hu,

            ct_image = to255(data)
            cv2.imwrite(train_image_path + '\\' + this_save_name, ct_image)
    print(order)
    order += 1






