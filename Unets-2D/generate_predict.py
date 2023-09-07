# -*- coding: utf-8 -*-
#!/usr/bin/python3
# -*- coding: utf-8 -*
from random import random

import torch
import time

import torchvision
from PIL.Image import NEAREST
from albumentations.pytorch.functional import img_to_tensor
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models import unet, nested_unet, loss_function, xnet, nfn_plus, unet_tiny
from datasets import lung
from torchvision import transforms
import os
from PIL import Image
import argparse
import numpy as np
from collections import OrderedDict
import cv2

# 选择网络模型

strat_time = time.time()
# net = unet.UNet(num_classes=2, in_channels=3)
save_name = ""
net = unet.UNet(num_classes=2, in_channels=3)
# 加载模型
ckpt = torch.load(r".pkl")
ckpt = ckpt['model_state_dict']
# new_state_dict = OrderedDict()
# for k, v in ckpt.items():
#     name = k[7:]  # remove `module.`，表面从第7个key值字符取到最后一个字符，正好去掉了module.
#     new_state_dict[name] = v  # 新字典的key值对应的value为一一对应的值。

net.load_state_dict(ckpt)
net.eval()

pre_data_path = r".../val/image"
dst_path = r"..." + "/" + save_name
os.makedirs(dst_path, exist_ok=True)

image_list = []
for file in os.listdir(pre_data_path):
    image_list.append(os.path.join(pre_data_path, file))

with torch.no_grad():
    i = 0
    for image in image_list:
        print(i)
        i += 1
        name = image.split("/")[-1]
        gt_mask = cv2.imread(image.replace('image', 'mask'), 0)
        if len(np.unique(gt_mask)) > 1:
            image = Image.open(image).convert('RGB')
            ori_size = image.size

            image = transforms.ToTensor()(image)
            # image = Variable(torch.unsqueeze(image, dim=0).float(), requires_grad=False)
            image = torch.unsqueeze(image, dim=0)
            outputs = net(image)
            outputs = outputs.squeeze(0)    # [2, 224, 224]

            if outputs.shape[0] == 2:
                mask = torch.max(outputs, 0)[1].cpu().numpy()
                a = np.zeros(mask.shape, np.uint8)
                a[mask == 1] = 255
                cat = np.hstack([gt_mask, a])
                print(dst_path + "/" + str(name), cat)
                #cv2.imwrite(dst_path + "/" + str(name), cat)
        else:
            pass


end_time = time.time()
seg_time = end_time - strat_time
print(save_name + "总耗时为：" + str(seg_time))


