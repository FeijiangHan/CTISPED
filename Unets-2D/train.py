#!/usr/bin/python3
# -*- coding: utf-8 -*
import os
# 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import time
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models import unet, nested_unet, loss_function, xnet, nfn_plus, lovasz_losses, unet_tiny
from datasets import lung
from torchvision import transforms
import os
import argparse
import numpy as np
from utils.metrics import compute_metrics
from utils.tools import create_directory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Parameter Setting
parser = argparse.ArgumentParser()
# Model Configuration
parser.add_argument('--model', type=str, default='unet', choices=['unet', 'r2unet', 'attention_unet', 'attention_r2unet', 'nested_unet',
                                    'xnet', 'nfn_plus'])
# Select DataSet 
parser.add_argument('--dataset', type=str, default='lung', choices=['lung'])
# Loss function
parser.add_argument('--loss', type=str, default='Dice', choices=['DiceBCE', 'CE', 'SCE', 'Dice', 'Lovasz'])
# tools.py: Improving with noisy
parser.add_argument('--noisy_rate', type=float, choices=[0.2, 0.3, 0.4])
parser.add_argument('--noisy_type', type=str, choices=['sy', 'asy'])
# Other configurations
parser.add_argument('--checkpoint', type=str, default=None)
parser.add_argument('--gpu', type=str, default='0', choices=['0', '1'])
parser.add_argument('--parallel', type=str, default='False', choices=['True', 'False'])
parser.add_argument('--num_workers', type=int, default=0, choices=list(range(17)))
parser.add_argument('--epoch', type=int, default=300)
parser.add_argument('--batch_size', type=int, default=2)
# Learning rate
parser.add_argument('--lr', type=float, default=1e-4)
# Learning rate decay
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--print_frequency', type=int, default=4)
parser.add_argument('--save_frequency', type=int, default=4)
args = parser.parse_args()

# Others
BASE_PATH = r'C:\Users\29185\Desktop'
format = '{}_{}_{}'.format(args.dataset, args.model, args.loss)
# ？
loss_weights = False
if loss_weights:
    format += "_unet_dice_debug"

log_path = os.path.join(BASE_PATH, 'log', format)
os.makedirs(log_path, exist_ok=True)
checkpoint_path_prefix = os.path.join(BASE_PATH, 'checkpoint', format)
os.makedirs(checkpoint_path_prefix, exist_ok=True)


# DEVICE = 'cuda'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loading Data
print('Loading data...')
# Choose the training data
if args.dataset == 'lung':
    dataset = lung.Lung
else:
    print('Data Abnormal!')
    pass

# resize the image and mask
transform = transforms.Compose([transforms.ToTensor()])
target_transform = transforms.Compose([transforms.ToTensor()])

# The training data and validation data paths
if args.dataset == 'lung':
    train_data = dataset(mode='train', transform=transform, target_transform=target_transform,
                         BASE_PATH=r"C:\Users\29185\Desktop\data\train")
    val_data = dataset(mode='val', transform=transform, target_transform=target_transform,
                       BASE_PATH=r"C:\Users\29185\Desktop\data\val")

train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True)
val_loader = DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
print('Create model...')

# Models selection
if args.model == 'unet':
    net = unet.UNet(num_classes=dataset.NUM_CLASSES, in_channels=dataset.CHANNELS_NUM)
elif args.model == 'r2unet':
    net = unet.UNet(num_classes=dataset.NUM_CLASSES, in_channels=dataset.CHANNELS_NUM, is_recurrent_residual=True)
elif args.model == 'attention_unet':
    net = unet.UNet(num_classes=dataset.NUM_CLASSES, in_channels=dataset.CHANNELS_NUM, is_attention=True)
elif args.model == 'attention_r2unet':
    net = unet.UNet(num_classes=dataset.NUM_CLASSES, in_channels=dataset.CHANNELS_NUM, is_attention=True,
                    is_recurrent_residual=True)
elif args.model == 'nested_unet':
    net = nested_unet.NestedUNet(num_classes=dataset.NUM_CLASSES, in_channels=dataset.CHANNELS_NUM)
elif args.model == 'xnet':
    net = xnet.XNet(num_classes=dataset.NUM_CLASSES, in_channels=dataset.CHANNELS_NUM)
elif args.model == 'nfn_plus':
    net = nfn_plus.NFNPlus(num_classes=dataset.NUM_CLASSES, in_channels=dataset.CHANNELS_NUM)
elif args.model == 'unet2tiny':
    net = unet_tiny.UNet2tiny(num_classes=dataset.NUM_CLASSES, in_channels=dataset.CHANNELS_NUM)
elif args.model == 'unet3tiny':
    net = unet_tiny.UNet3tiny(num_classes=dataset.NUM_CLASSES, in_channels=dataset.CHANNELS_NUM)
elif args.model == 'unet1tiny':
    net = unet_tiny.UNet1tiny(num_classes=dataset.NUM_CLASSES, in_channels=dataset.CHANNELS_NUM)

# Set the optimizer and loss function
# optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

# Select the loss function
if args.loss == 'DiceBCE':
    criterion = loss_function.DiceAndBCELoss(dataset.NUM_CLASSES)
elif args.loss == 'CE':
    criterion = torch.nn.CrossEntropyLoss()
elif args.loss == 'SCE':
    criterion = loss_function.SCELoss(dataset.NUM_CLASSES, alpha=1, beta=1)
elif args.loss == 'Dice':
    criterion = loss_function.SoftDiceLoss(dataset.NUM_CLASSES)
elif args.loss == 'Lovasz':
    criterion = loss_function.LovaszLoss()


print('<================== Parameters ==================>')
print('model: {}'.format(net))
print('dataset: {}(training={}, validation={})'.format(train_data, len(train_data), len(val_data)))
print('batch_size: {}'.format(args.batch_size))
print('batch_num: {}'.format(len(train_loader)))
print('epoch: {}'.format(args.epoch))
print('loss_function: {}'.format(criterion))
print('optimizer: {}'.format(optimizer))
print('tensorboard_log_path: {}'.format(log_path))
print('<================================================>')


# If you want to use multiple GPU
if args.parallel == 'True':
    print('Use DataParallel.')
    net = torch.nn.DataParallel(net)
# GPU or CPU
net = net.to(device)

start_epoch = 0
temp = 0
# Loading model
if args.checkpoint is not None:
    checkpoint_data = torch.load(args.checkpoint)
    print('**** Load model and optimizer data from {} ****'.format(args.checkpoint))

    # loading the data of modal and optimizer
    net.load_state_dict(checkpoint_data['model_state_dict'])
    optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])

    # To load the last epoch trained last time and the last temp printed, 
    # add 1 to the previous one as a starting point
    start_epoch = checkpoint_data['epoch'] + 1
    temp = checkpoint_data['temp'] + 1

    args.epoch += start_epoch
    # temp = (len(train_loader) // args.print_frequency) * start_epoch + 1

else:
    # Remove and reconstruct the previous log and training again
    create_directory(log_path)

writer = SummaryWriter(log_dir=log_path, flush_secs=30)
# The process of training and validating
print('Start training...')
for epoch in range(start_epoch, args.epoch):
    # 训练
    loss_all = []
    predictions_all = []
    labels_all = []
    print('-------------------------------------- Training {} --------------------------------------'.format(epoch + 1))
    net.train()
    for index, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)   #
        loss = 0
        # 如果使用deep supervision，返回1个list（包含多个输出），计算每个输出的loss，最后求平均
        if isinstance(outputs, list):
            for out in outputs:
                loss += criterion(out, labels.long())
            loss /= len(outputs)
        else:
            loss = criterion(outputs, labels.long())
        # 计算在该批次上的平均损失函数 (average loss)
        loss /= inputs.size(0)

        # 更新网络参数 (update network parameters)
        loss.backward()
        optimizer.step()

        loss_all.append(loss.item())

        if isinstance(outputs, list):
            # 若使用deep supervision，用最后的输出来进行预测
            # If deep supervision is used, the final output is used to make predictions
            predictions = torch.max(outputs[-1], dim=1)[1].cpu().numpy().astype(np.int)
        else:
            # 将概率最大的类别作为预测的类别
            # Take the category with the highest probability as the category of prediction
            predictions = torch.max(outputs, dim=1)[1].cpu().numpy().astype(np.int)

        labels = labels.cpu().numpy().astype(np.int)

        predictions_all.append(predictions)
        labels_all.append(labels)

        if (index + 1) % args.print_frequency == 0:
            # 计算打印间隔的平均损失函数
            # Calculate the average loss function of the print interval
            avg_loss = np.mean(loss_all)
            loss_all = []

            writer.add_scalar('train/loss', avg_loss, temp)
            temp += 1

            print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} ".format(
                epoch + 1, args.epoch, index + 1, len(train_loader), avg_loss))

    # 使用混淆矩阵计算语义分割中的指标
    # Calculate indexes in semantic segmentation using confusion matrix
    iou, miou, dsc, mdsc, ac, pc, mpc, se, mse, sp, msp, f1, mf1 = compute_metrics(predictions_all, labels_all,
                                                                                   dataset.NUM_CLASSES)

    writer.add_scalars('train/metrics', dict(miou=miou, mdsc=mdsc, mpc=mpc, ac=ac, mse=mse, msp=msp, mf1=mf1), epoch)

    print('Training: MIoU: {:.4f}, MDSC: {:.4f}, MPC: {:.4f}, AC: {:.4f}, MSE: {:.4f}, MSP: {:.4f}, MF1: {:.4f}'.format(
        miou, mdsc, mpc, ac, mse, msp, mf1
    ))

    # 验证 (validation)
    loss_all = []
    predictions_all = []
    labels_all = []

    print('-------------------------------------- Validation {} ------------------------------------'.format(epoch + 1))

    net.eval()
    with torch.no_grad():
        for _, data in enumerate(val_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = net(inputs)

            loss = 0
            # 如果使用deep supervision，返回1个list（包含多个输出），计算每个输出的loss，最后求平均
            # If you use deep supervision, return a list (containing multiple outputs), 
            # calculate the loss for each output, and finally average
            if isinstance(outputs, list):
                for out in outputs:
                    loss += criterion(out, labels.long())
                loss /= len(outputs)
            else:
                loss = criterion(outputs, labels.long())
            # 计算在该批次上的平均损失函数
            # Calculate the average loss function on the batch
            loss /= inputs.size(0)

            loss_all.append(loss.item())

            if isinstance(outputs, list):
                # 若使用deep supervision，用最后一个输出来进行预测
                # If deep supervision is used, use the last output to make the prediction
                predictions = torch.max(outputs[-1], dim=1)[1].cpu().numpy().astype(np.int)
            else:
                # 将概率最大的类别作为预测的类别
                # Take the category with the highest probability as the category of prediction
                predictions = torch.max(outputs, dim=1)[1].cpu().numpy().astype(np.int)
            labels = labels.cpu().numpy().astype(np.int)

            predictions_all.append(predictions)
            labels_all.append(labels)

    # 使用混淆矩阵计算语义分割中的指标
    # Calculate indexes in semantic segmentation using confusion matrix
    iou, miou, dsc, mdsc, ac, pc, mpc, se, mse, sp, msp, f1, mf1 = compute_metrics(predictions_all, labels_all,
                                                                                   dataset.NUM_CLASSES)
    avg_loss = np.mean(loss_all)

    writer.add_scalar('val/loss', avg_loss, epoch)
    writer.add_scalars('val/metrics', dict(miou=miou, mdsc=mdsc, mpc=mpc, ac=ac, mse=mse, msp=msp, mf1=mf1), epoch)

    # 绘制每个类别的IoU
    # Draw the IoU for each category
    temp_dict = {'miou': miou}
    for i in range(dataset.NUM_CLASSES):
        temp_dict['class{}'.format(i)] = iou[i]
    writer.add_scalars('val/class_iou', temp_dict, epoch)

    print('Training: MIoU: {:.4f}, MDSC: {:.4f}, MPC: {:.4f}, AC: {:.4f}, MSE: {:.4f}, MSP: {:.4f}, MF1: {:.4f}'.format(
        miou, mdsc, mpc, ac, mse, msp, mf1
    ))

    # 保存模型参数和优化器参数
    # Save model parameters and optimizer parameters
    if (epoch + 1) % args.save_frequency == 0:
        checkpoint_path = '{}_{}_{}.pkl'.format(format, time.strftime('%m%d_%H%M', time.localtime()), epoch)
        # save_checkpoint_path = checkpoint_path_prefix + '/' + checkpoint_path
        save_checkpoint_path = os.path.join(checkpoint_path_prefix, checkpoint_path)
        torch.save({
            'is_parallel': args.parallel,
            'epoch': epoch,
            'temp': temp,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()},
            save_checkpoint_path)
        print('Save model at {}.'.format(save_checkpoint_path))
