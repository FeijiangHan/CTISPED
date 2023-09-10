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

# Select the network model
# net = xnet.XNet(num_classes=2, in_channels=3)
net = unet.UNet(num_classes=2, in_channels=3)
# net = unet.UNet(num_classes=2, in_channels=3, is_attention=True)

# Load the model
ckpt = torch.load(r".pkl")
ckpt = ckpt['model_state_dict']
new_state_dict = OrderedDict()
for k, v in ckpt.items():
    # name = k[7:]  # remove `module.`, get from 7th char to last char, remove module. 
    new_state_dict[k] = v  # new dict's key value is the corresponding value.

net.load_state_dict(new_state_dict)
net.eval()

pre_data_path = r"...image" 
dst_path = r"...preds"
os.makedirs(dst_path, exist_ok=True)
image_list = []
for file in os.listdir(pre_data_path):
    image_list.append(os.path.join(pre_data_path, file))
    
# Validation   
loss_all = []
predictions_all = []
labels_all = []

with torch.no_grad():
    i = 0
    for image in image_list:
        print(i) 
        i += 1
        name = image.split("\\")[-1]
        
        # Read original image
        org_image = cv2.imread(image, 0)
        org_mask = cv2.imread(image.replace('image', 'label'), 0)
        
        # Create labels array
        labels = np.zeros(org_mask.shape, np.uint8)
        labels[org_mask != 0] = 1
        
        # Preprocess image 
        image = Image.open(image).convert('RGB')
        ori_size = image.size
        image = transforms.ToTensor()(image)
        
        image = Variable(torch.unsqueeze(image, dim=0).float(), requires_grad=False)
        outputs = net(image)    
        
        # Get predictions
        if isinstance(outputs, list):
            # Use last output for prediction if deep supervision
            predictions = torch.max(outputs[-1], dim=1)[1].cpu().numpy().astype(np.int)
        else:
            predictions = torch.max(outputs, dim=1)[1].cpu().numpy().astype(np.int)
            
        labels = labels.astype(np.int)
        predictions_all.append(predictions)
        labels_all.append(labels)
        
        outputs = outputs.squeeze(0)    
        mask = torch.max(outputs, 0)[1].cpu().numpy()
        a = np.zeros(mask.shape, np.float32)
        a[mask == 1] = 255
        
        # Concatenate images for display
        cat_img = np.hstack([org_image, org_mask, a])
        print(dst_path + '\\' + str(name), cat_img)
        
    # Compute segmentation metrics
    iou, miou, dsc, mdsc, ac, pc, mpc, se, mse, sp, msp, f1, mf1 = compute_metrics(predictions_all, labels_all,num_classes=2)

    print('Training: MIoU: {:.4f}, MDSC: {:.4f}, MPC: {:.4f}, AC: {:.4f}, MSE: {:.4f}, MSP: {:.4f}, MF1: {:.4f}'.format(
        miou, mdsc, mpc, ac, mse, msp, mf1
    ))