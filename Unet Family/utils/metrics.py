#!/usr/bin/python3
# -*- coding: utf-8 -*
import numpy as np


def compute_confusion_matrix(predictions_all, labels_all, num_classes):
    """根据每批数据的预测和原始的label，计算混淆矩阵"""

    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int)
    for predictions, labels in zip(predictions_all, labels_all):
        predictions = predictions.flatten()
        labels = labels.flatten()

        # np.bincount统计每个值的出现次数
        temp = np.bincount(num_classes * labels + predictions, minlength=num_classes ** 2).reshape(
            (num_classes, num_classes))

        confusion_matrix += temp

    return confusion_matrix


def compute_metrics(predictions_all, labels_all, num_classes):
    """计算语义分割中的指标
    https://github.com/ZijunDeng/pytorch-semantic-segmentation/blob/master/utils/misc.py，
    https://github.com/LeeJunHyun/Image_Segmentation,
    https://www.cnblogs.com/Trevo/p/11795503.html，
    https://www.aiuai.cn/aifarm1330.html"""

    confusion_matrix = compute_confusion_matrix(predictions_all, labels_all, num_classes)

    # 每个类别的交并比和平均的交并比
    # IoU=TP/(TP+FP+FN)
    iou = np.diag(confusion_matrix) / (
            confusion_matrix.sum(axis=1) + confusion_matrix.sum(axis=0) - np.diag(confusion_matrix))
    miou = np.nanmean(iou)

    # 每个类别的Dice相似系数和平均的Dice相似系数
    # DSC=2*TP/(TP+FN+TP+FP)
    dsc = 2 * np.diag(confusion_matrix) / (confusion_matrix.sum(axis=1) + confusion_matrix.sum(axis=0))
    mdsc = np.nanmean(dsc)

    # 所有像素总的准确率、每个类别的像素准确率、平均的像素准确率
    # AC=(TP+TN)/(TP+TN+FP+FN)
    # PC=TP/(TP+FP)
    ac = np.diag(confusion_matrix).sum() / confusion_matrix.sum()
    pc = np.diag(confusion_matrix) / confusion_matrix.sum(axis=0)
    mpc = np.nanmean(pc)

    # 敏感性，Sensitivity
    # SE=TP/(TP+FN)
    se = np.diag(confusion_matrix) / confusion_matrix.sum(axis=1)
    mse = np.nanmean(se)

    # 特异性，Specificity
    # SP=TN/(TN+FP)
    sp = [(np.diag(confusion_matrix).sum() - confusion_matrix[i, i]) /
          (np.diag(confusion_matrix).sum() + confusion_matrix.sum(axis=0)[i] - 2 * confusion_matrix[i, i])
          for i in range(len(confusion_matrix))]
    msp = np.nanmean(sp)

    # F1分数，F1 score
    # f1=2*PC*SE/(PC+SE)
    f1 = 2 * pc * se / (pc + se)
    mf1 = np.nanmean(f1)

    print('Confusion Matrix:')
    print(confusion_matrix)
    print('IoU: {}'.format(iou))
    print('DSC: {}'.format(dsc))
    print('PC: {}'.format(pc))
    print('SE: {}'.format(se))
    print('SP: {}'.format(sp))
    print('F1: {}'.format(f1))

    return iou, miou, dsc, mdsc, ac, pc, mpc, se, mse, sp, msp, f1, mf1
