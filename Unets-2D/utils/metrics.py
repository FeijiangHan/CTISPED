#!/usr/bin/python3
# -*- coding: utf-8 -*
import numpy as np

def compute_confusion_matrix(predictions_all, labels_all, num_classes):
    """Compute confusion matrix based on predictions and labels of each batch"""

    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int)
    for predictions, labels in zip(predictions_all, labels_all):
        predictions = predictions.flatten() 
        labels = labels.flatten()

        # np.bincount counts frequency of each value
        temp = np.bincount(num_classes * labels + predictions, minlength=num_classes**2).reshape(
            (num_classes, num_classes))

        confusion_matrix += temp

    return confusion_matrix

def compute_metrics(predictions_all, labels_all, num_classes):
    """Compute segmentation metrics
    
    References:
    https://github.com/ZijunDeng/pytorch-semantic-segmentation/blob/master/utils/misc.py
    https://github.com/LeeJunHyun/Image_Segmentation
    https://www.cnblogs.com/Trevo/p/11795503.html
    https://www.aiuai.cn/aifarm1330.html"""
    
    confusion_matrix = compute_confusion_matrix(predictions_all, labels_all, num_classes)

    # IoU and mean IoU 
    # IoU = TP/(TP+FP+FN)
    iou = np.diag(confusion_matrix) / (
            confusion_matrix.sum(axis=1) + confusion_matrix.sum(axis=0) - np.diag(confusion_matrix))
    miou = np.nanmean(iou)

    # DSC and mean DSC
    # DSC = 2*TP/(TP+FN+TP+FP)
    dsc = 2 * np.diag(confusion_matrix) / (confusion_matrix.sum(axis=1) + confusion_matrix.sum(axis=0))
    mdsc = np.nanmean(dsc)

    # Overall pixel accuracy, class pixel accuracy, mean pixel accuracy
    # AC = (TP+TN)/(TP+TN+FP+FN)
    # PC = TP/(TP+FP)
    ac = np.diag(confusion_matrix).sum() / confusion_matrix.sum()
    pc = np.diag(confusion_matrix) / confusion_matrix.sum(axis=0)
    mpc = np.nanmean(pc)

    # Sensitivity 
    # SE = TP/(TP+FN)
    se = np.diag(confusion_matrix) / confusion_matrix.sum(axis=1)
    mse = np.nanmean(se)

    # Specificity
    # SP = TN/(TN+FP)
    sp = [(np.diag(confusion_matrix).sum() - confusion_matrix[i, i]) / 
          (np.diag(confusion_matrix).sum() + confusion_matrix.sum(axis=0)[i] - 2 * confusion_matrix[i, i])
          for i in range(len(confusion_matrix))]
    msp = np.nanmean(sp)

    # F1 score and mean F1 score
    # F1 = 2*PC*SE/(PC+SE)
    f1 = 2 * pc * se / (pc + se)
    mf1 = np.nanmean(f1)

    print('Confusion Matrix:')
    print(confusion_matrix)
    print('IoU:', iou)
    print('DSC:', dsc)
    print('PC:', pc) 
    print('SE:', se)
    print('SP:', sp)
    print('F1:', f1)

    return iou, miou, dsc, mdsc, ac, pc, mpc, se, mse, sp, msp, f1, mf1