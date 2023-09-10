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
    """Generate indexes of images to add noise"""
    
    num = int(dataset_size * noisy_rate)
    random.seed(seed)  
    indexes = random.sample(range(0, dataset_size), num)
    return indexes

def add_noisy_triangle_label(mask):
    """Add noisy triangle annotation on the mask"""
    
    while True:
        noisy_image = Image.fromarray(np.zeros(mask.size, dtype=np.uint8))
        # Generate random triangle vertex coordinates 
        noisy_points = np.random.randint(0, 256, (3, 2))
        
        # Triangle span cannot be too large
        if (noisy_points.max(axis=0) - noisy_points.min(axis=0)).max() > 75:
            continue
            
        noisy_points = noisy_points.flatten()
        noisy_points = tuple(noisy_points)
        ImageDraw.Draw(noisy_image).polygon(noisy_points, fill=255)

        # Noisy label area cannot be too large
        noisy_array = np.asarray(noisy_image) / 255
        if noisy_array.sum() <= 1500 and noisy_array.sum() >= 1000:
            # Add noise to original mask
            ImageDraw.Draw(mask).polygon(noisy_points, fill=255)
            break

    return mask

def create_directory(dir_path):
    """Create directory, delete existing one if needed"""
    
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        print('Removed {}'.format(dir_path))
    os.mkdir(dir_path)
    print('Created {}'.format(dir_path))

def process_binary_mask_tensor(mask):
    """Process generated Tensor mask"""
    
    assert isinstance(mask, torch.Tensor), 'Input is not Tensor'

    # ToTensor adds a channel dim, remove for label 
    if len(mask.shape) == 3:
        mask = torch.squeeze(mask, dim=0)

    mask = mask.to(torch.uint8)
    mask[mask != 0] = 1

    return mask

def process_multiple_mask_tensor(mask, color_map):
    """Process generated Tensor mask with multiple classes"""

    assert isinstance(mask, torch.Tensor), 'Input is not Tensor'

    # ToTensor adds a channel dim, remove for label
    if len(mask.shape) == 3:
        mask = torch.squeeze(mask, dim=0)

    # If mask resized, modify noise and convert to uint8
    if torch.max(mask).item() <= 1:
        # Restore from normalization 
        mask = (mask * 255).to(torch.uint8)

        # Record noise coordinate 
        noise_indexes = torch.ones_like(mask, dtype=torch.bool)

        # Remove noise
        for color_value, class_id in color_map.items():
            noise_indexes = noise_indexes & (mask != color_value)
        mask[noise_indexes] = 0

        # Replace color values with class ids
        for color_value, class_id in color_map.items():
            mask[mask == color_value] = class_id
            
        # Filter small spurious labels
        for class_id in color_map.values():
            if torch.sum(mask == class_id).item() <= 10:
                mask[mask == class_id] = 0

    return mask
    
def is_augmented(path):
    """Check if path is for augmented data"""
    
    augmented_types = ['VerticalFlip', 'HorizontalFlip', 'Transpose', 'RandomRotate90']
    for t in augmented_types:
        if path.__contains__(t):
            return True
    return False

def remove_augmented_item(old_items):
    """Remove augmented data items"""
    
    new_items = []
    for item in old_items:
        if not is_augmented(item['image_path']):
            new_items.append(item)
    return new_items

def make_dataset(mode, base_path, is_contain_augmented_data):
    """Load json files for training/validation"""

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
    """Initialize weights in the model"""

    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.BatchNorm2d):
            module.weight.data.fill_(1)
            module.bias.data.zero_()