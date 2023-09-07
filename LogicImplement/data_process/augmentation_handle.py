# -*- coding: utf-8 -*-

import random

import cv2
import numpy as np
import os.path
import copy


# 椒盐噪声
def SaltAndPepper(src, percetage=0.2):
    SP_NoiseImg = src.copy()
    SP_NoiseNum = int(percetage * src.shape[0] * src.shape[1])
    for i in range(SP_NoiseNum):
        randR = np.random.randint(0, src.shape[0] - 1)
        randG = np.random.randint(0, src.shape[1] - 1)
        randB = np.random.randint(0, 3)
        if np.random.randint(0, 1) == 0:
            SP_NoiseImg[randR, randG, randB] = 0
        else:
            SP_NoiseImg[randR, randG, randB] = 255
    return SP_NoiseImg


# 高斯噪声
def addGaussianNoise(image, percetage=0.2):
    G_Noiseimg = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    G_NoiseNum = int(percetage * image.shape[0] * image.shape[1])
    for i in range(G_NoiseNum):
        temp_x = np.random.randint(0, h)
        temp_y = np.random.randint(0, w)
        G_Noiseimg[temp_x][temp_y][np.random.randint(3)] = np.random.randn(1)[0]
    return G_Noiseimg


# 昏暗
def darker(image, percetage=0.9):
    image_copy = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    # get darker
    for xi in range(0, w):
        for xj in range(0, h):
            image_copy[xj, xi, 0] = int(image[xj, xi, 0] * percetage)
            image_copy[xj, xi, 1] = int(image[xj, xi, 1] * percetage)
            image_copy[xj, xi, 2] = int(image[xj, xi, 2] * percetage)
    return image_copy


# 亮度
def brighter(image, percetage=1.5):
    image_copy = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    # get brighter
    for xi in range(0, w):
        for xj in range(0, h):
            image_copy[xj, xi, 0] = np.clip(int(image[xj, xi, 0] * percetage), a_max=255, a_min=0)
            image_copy[xj, xi, 1] = np.clip(int(image[xj, xi, 1] * percetage), a_max=255, a_min=0)
            image_copy[xj, xi, 2] = np.clip(int(image[xj, xi, 2] * percetage), a_max=255, a_min=0)
    return image_copy


# 旋转
def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    # If no rotation center is specified, the center of the image is set as the rotation center
    if center is None:
        center = (w / 2, h / 2)
    m = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, m, (w, h))
    return rotated


# 翻转
def flip(image):
    flipped_image = np.fliplr(image)
    return flipped_image


# 图片文件夹路径
image_dir = r'C:\data\TianChiCTSeg\data\train\image\\'
mask_dir = r'C:\data\TianChiCTSeg\data\train\mask\\'
i = 0
j = 0
for img_name in os.listdir(image_dir):
    num = random.randint(0, 10)
    # print("num:", num)
    i += 1
    if num == 2 or num == 3:
        j += 1
        print("aug")
        img_path = image_dir + img_name
        mask_path = mask_dir + img_name
        print(img_path)
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path)

        rotated_90_image = rotate(img, 90)
        rotated_90_mask = rotate(mask, 90)
        cv2.imwrite(image_dir + img_name[0:-4] + '_r90.png', rotated_90_image)
        cv2.imwrite(mask_dir.replace("image", "mask") + img_name[0:-4] + '_r90.png', rotated_90_mask)

        brighter_image = brighter(img)
        # brighter_mask = brighter(mask)
        cv2.imwrite(image_dir + img_name[0:-4] + '_brighter.png', brighter_image)
        cv2.imwrite(mask_dir.replace("image", "mask") + img_name[0:-4] + '_brighter.png', mask)

        darker_image = darker(img)
        # darker_mask = darker(mask)
        cv2.imwrite(image_dir + img_name[0:-4] + '_darker.png', darker_image)
        cv2.imwrite(mask_dir.replace("image", "mask") + img_name[0:-4] + '_darker.png', mask)

        addGaussianNoise_image = addGaussianNoise(img, 90)
        addGaussianNoise_mask = addGaussianNoise(mask, 90)
        cv2.imwrite(image_dir + img_name[0:-4] + '_addGaussianNoise.png', addGaussianNoise_image)
        cv2.imwrite(mask_dir.replace("image", "mask") + img_name[0:-4] + '_addGaussianNoise.png', addGaussianNoise_mask)
        #
        SaltAndPepper_image = SaltAndPepper(img, 90)
        SaltAndPepper_mask = SaltAndPepper(mask, 90)
        cv2.imwrite(image_dir + img_name[0:-4] + '_SaltAndPepper.png', SaltAndPepper_image)
        cv2.imwrite(mask_dir.replace("image", "mask") + img_name[0:-4] + '_SaltAndPepper.png', SaltAndPepper_mask)
    else:
        print("不aug")
    print(f"进行到{i}张")
    print(f"已增强{j}张")
