# -*- coding: utf-8 -*-
#!/usr/bin/python3
# -*- coding: utf-8 -*
from random import random

import torch
from torch.autograd import Variable
from models import unet
from torchvision import transforms
import os
from PIL import Image
import argparse
import numpy as np
from collections import OrderedDict
import cv2
from utils.metrics import compute_metrics

# 选择网络模型
# net = xnet.XNet(num_classes=2, in_channels=3)
net = unet.UNet(num_classes=2, in_channels=3)
# net = unet.UNet(num_classes=2, in_channels=3, is_attention=True)

# 加载模型
ckpt = torch.load(r".pkl")
ckpt = ckpt['model_state_dict']
new_state_dict = OrderedDict()
for k, v in ckpt.items():
    # name = k[7:]  # remove `module.`，表面从第7个key值字符取到最后一个字符，正好去掉了module.
    new_state_dict[k] = v  # 新字典的key值对应的value为一一对应的值。

net.load_state_dict(new_state_dict)
net.eval()

pre_data_path = r"...image"
dst_path = r"...preds"
os.makedirs(dst_path, exist_ok=True)
image_list = []
for file in os.listdir(pre_data_path):
    image_list.append(os.path.join(pre_data_path, file))

# 验证
loss_all = []
predictions_all = []
labels_all = []


with torch.no_grad():
    i = 0
    for image in image_list:
        print(i)
        i += 1
        name = image.split("\\")[-1]
        # print(name)
        # 读取原图
        org_image = cv2.imread(image, 0)
        org_mask = cv2.imread(image.replace('image', 'label'), 0)

        # mask = Image.open(image.replace('image', 'label')).convert('L')
        labels = np.zeros(org_mask.shape, np.uint8)
        labels[org_mask != 0] = 1

        image = Image.open(image).convert('RGB')
        ori_size = image.size
        # image = image.resize((224, 224), resample=NEAREST)
        # image = img_to_tensor(image)
        image = transforms.ToTensor()(image)
        # print(image.shape)  # [3, 224, 224]
        image = Variable(torch.unsqueeze(image, dim=0).float(), requires_grad=False)
        # print(image.shape)  # [1, 3, 224, 224]
        outputs = net(image)    # # [1, 2, 224, 224], 此2应该为类别数

        if isinstance(outputs, list):
            # 若使用deep supervision，用最后一个输出来进行预测
            predictions = torch.max(outputs[-1], dim=1)[1].cpu().numpy().astype(np.int)
        else:
            # 将概率最大的类别作为预测的类别
            predictions = torch.max(outputs, dim=1)[1].cpu().numpy().astype(np.int)
        labels = labels.astype(np.int)

        predictions_all.append(predictions)
        labels_all.append(labels)
        # softmax = torch.nn.Softmax(dim=1)
        # outputs = softmax(outputs)
        outputs = outputs.squeeze(0)    # [2, 224, 224]

        # mask = np.argmax(outputs, axis=0)
        mask = torch.max(outputs, 0)[1].cpu().numpy()
        # print(mask.max())
        # print(mask.shape)
        a = np.zeros(mask.shape, np.float32)
        a[mask == 1] = 255
        # a[mask == 0] = 0
        # for i in range(np.shape(a)[0]):
        im = Image.fromarray(np.uint8(a[:, :]))
        # 下面进行拼接展示

        cat_img = np.hstack([org_image, org_mask, a])
        print(dst_path + '\\' + str(name), cat_img)
        ##cv2.imwrite(dst_path + '\\' + str(name), cat_img)


    # 使用混淆矩阵计算语义分割中的指标
    iou, miou, dsc, mdsc, ac, pc, mpc, se, mse, sp, msp, f1, mf1 = compute_metrics(predictions_all, labels_all,
                                                                                   num_classes=2)
    print('Training: MIoU: {:.4f}, MDSC: {:.4f}, MPC: {:.4f}, AC: {:.4f}, MSE: {:.4f}, MSP: {:.4f}, MF1: {:.4f}'.format(
        miou, mdsc, mpc, ac, mse, msp, mf1
    ))
        # im = im.resize(ori_size, resample=NEAREST)     # 需要使用最近邻插值
        # im.convert('RGB').save(dst_path + "/" + str(name))


