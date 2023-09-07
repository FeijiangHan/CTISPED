import os
import pandas as pd
import numpy as np
import math
import copy
import matplotlib.pyplot as plt
import sklearn.metrics as skl_metrics
from matplotlib.ticker import FixedFormatter
import SimpleITK as sitk
import skimage.measure
from skimage.morphology import remove_small_objects
from scipy.ndimage.morphology import binary_dilation
import csv


def generate_binary_structure(size, distance, weight=[1, 1, 1]):
    shift_z = (np.arange(0, size)) - size // 2
    shift_y = (np.arange(0, size)) - size // 2
    shift_x = (np.arange(0, size)) - size // 2
    shift_z, shift_y, shift_x = np.meshgrid(shift_z, shift_y, shift_x, indexing='ij')
    shifts = np.stack((shift_z, shift_y, shift_x), axis=-1)
    shifts = ((shifts * np.array(weight)) ** 2).sum(axis=-1) ** 0.5
    return shifts <= distance


def resample_simg(simg, new_spacing=(1.0, 1.0, 1.0), interp=sitk.sitkBSpline):
    new_spacing = new_spacing[::-1]
    identity1 = sitk.Transform(3, sitk.sitkIdentity)

    sp1 = simg.GetSpacing()
    sz1 = simg.GetSize()
    sz2 = (int(round(sz1[0] * sp1[0] / new_spacing[0])), int(round(sz1[1] * sp1[1] / new_spacing[1])),
           int(round(sz1[2] * sp1[2] / new_spacing[2])))

    new_origin = simg.GetOrigin()
    new_origin = (new_origin[0] - sp1[0] / 2 + new_spacing[0] / 2, new_origin[1] - sp1[1] / 2 + new_spacing[1] / 2,
                  new_origin[2] - sp1[2] / 2 + new_spacing[2] / 2)
    imRefImage = sitk.Image(sz2, simg.GetPixelIDValue())
    imRefImage.SetSpacing(new_spacing)
    imRefImage.SetOrigin(new_origin)
    imRefImage.SetDirection(simg.GetDirection())
    resampled_image = sitk.Resample(simg, imRefImage, identity1, interp)
    return resampled_image


def redirection_simg(itkimg, target_direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)):
    # target direction should be orthognal, i.e. (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

    # permute axis
    tmp_target_direction = np.abs(np.round(np.array(target_direction))).reshape(3, 3).T
    current_direction = np.abs(np.round(itkimg.GetDirection())).reshape(3, 3).T

    permute_order = []
    if not np.array_equal(tmp_target_direction, current_direction):
        for i in range(3):
            for j in range(3):
                if np.array_equal(tmp_target_direction[i], current_direction[j]):
                    permute_order.append(j)
                    break
        redirect_img = sitk.PermuteAxes(itkimg, permute_order)
    else:
        redirect_img = itkimg
    # flip axis
    current_direction = np.round(np.array(redirect_img.GetDirection())).reshape(3, 3).T
    current_direction = np.max(current_direction, axis=1)

    tmp_target_direction = np.array(target_direction).reshape(3, 3).T
    tmp_target_direction = np.max(tmp_target_direction, axis=1)
    flip_order = ((tmp_target_direction * current_direction) != 1)
    fliped_img = sitk.Flip(redirect_img, [bool(flip_order[0]), bool(flip_order[1]), bool(flip_order[2])])

    return fliped_img


def computeFROC(FROCGTList, FROCProbList, totalNumberOfImages, excludeList):
    # Remove excluded candidates
    FROCGTList_local = []
    FROCProbList_local = []
    for i in range(len(excludeList)):
        if excludeList[i] == False:
            FROCGTList_local.append(FROCGTList[i])
            FROCProbList_local.append(FROCProbList[i])

    numberOfDetectedLesions = sum(FROCGTList_local)
    totalNumberOfLesions = sum(FROCGTList)
    totalNumberOfCandidates = len(FROCProbList_local)
    fpr, tpr, thresholds = skl_metrics.roc_curve(FROCGTList_local, FROCProbList_local)
    if sum(FROCGTList) == len(
            FROCGTList):  # Handle border case when there are no false positives and ROC analysis give nan values.
        print("WARNING, this system has no false positives..")
        fps = np.zeros(len(fpr))
    else:
        fps = fpr * (totalNumberOfCandidates - numberOfDetectedLesions) / totalNumberOfImages
    sens = (tpr * numberOfDetectedLesions) / totalNumberOfLesions
    return fps, sens, thresholds


def plot_FROC(fps, sens, thresholds, key_pt=[1.0 / 8, 1.0 / 4, 1.0 / 2, 1.0, 2.0, 4.0, 8.0], save_path=None,
              title='FROC'):
    FROC_minX = key_pt[0]
    FROC_maxX = key_pt[-1]
    bLogPlot = False

    valid_n = (thresholds > 0).sum()
    fps_itp = np.linspace(FROC_minX, FROC_maxX, num=10001)
    sens_itp = np.interp(fps_itp, fps[:valid_n], sens[:valid_n])
    froc_score = 0.0
    for k in key_pt:
        val_idxes = np.where(np.abs(fps_itp - k) <= 1e-2)
        froc_score += np.mean(sens_itp[val_idxes])
    froc_score /= len(key_pt)
    print('FROC score {}'.format(froc_score))

    fig1 = plt.figure()
    ax = plt.gca()
    clr = 'b'
    plt.plot(fps_itp, sens_itp, color=clr, label="%s" % "FROC", lw=2)
    xmin = FROC_minX
    xmax = FROC_maxX
    plt.xlim(xmin, xmax)
    plt.ylim(0, 1)
    plt.xlabel('Average number of false positives per scan')
    plt.ylabel('Sensitivity')
    plt.legend(loc='lower right')
    plt.title(title)

    if bLogPlot:
        plt.xscale('log', basex=2)
        ax.xaxis.set_major_formatter(FixedFormatter(key_pt))

    # set your ticks manually
    ax.xaxis.set_ticks(key_pt)
    ax.yaxis.set_ticks(np.arange(0, 1.1, 0.1))
    plt.grid(b=True, which='both')
    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, 'froc_plot.png'), dpi=600)
    plt.show()


def dice_score(pred, target, eps=1e-4):
    tp = np.sum(pred * target)
    p = np.sum(pred)
    t = np.sum(target)
    score = tp / (p + t + eps)
    return score


def fracCADEvaluation(gt_list, pred_list, pred_info_list, title, eps=0.):
    matched_overlap = 0
    min_prob = 0.1

    candTPs = 0
    candFPs = 0
    candFNs = 0
    candPs = 0
    doubleDetection = 0
    minProbValue = -1000000000.0  # minimum value of a float
    FROCGTList = []
    FROCProbList = []
    FPDivisorList = []
    excludeList = []
    patient_num = 0

    for gt_path, pred_path, pred_info_path in zip(gt_list, pred_list, pred_info_list):
        # if '040RefStd' not in gt_path:
        #     continue
        patient_num += 1
        slabel = sitk.ReadImage(gt_path)
        spred = sitk.ReadImage(pred_path)
        pred_info = np.array(pd.read_csv(pred_info_path))
        if pred_info.shape[0] == 0:
            pred_info = np.array([[1, 0., 0., 0., 0.]])

        if eps > 0:
            spacing = slabel.GetSpacing()
            kernal_size = [math.ceil(eps / spacing[0]), math.ceil(eps / spacing[1]), math.ceil(eps / spacing[2])]
            dilate_filter = sitk.BinaryDilateImageFilter()
            dilate_filter.SetKernelRadius(kernal_size)
            dilate_filter.SetKernelType(sitk.sitkBall)
            slabel = dilate_filter.Execute(slabel > 0)
            slabel = sitk.ConnectedComponent(slabel, True)

        gt_array = sitk.GetArrayFromImage(slabel)
        pred_array = sitk.GetArrayFromImage(spred)

        assert gt_array.shape == pred_array.shape

        # if eps > 0:
        # kernal_size = np.ceil(eps / np.min(spacing)) * 2 + 1
        # struct = generate_binary_structure(size=int(kernal_size), distance=eps, weight=spacing)
        # gt_array = np.clip(gt_array, a_min=0, a_max=1)
        # gt_array = binary_dilation(gt_array, structure=struct, iterations=1)
        # gt_array = skimage.measure.label(gt_array, connectivity=3)

        num_gt = gt_array.max()
        num_pred = int(pred_info[:, 0].max())

        unmatched_pred = list(range(1, num_pred + 1))
        matched_gt = []

        candPs += (pred_info[:, 0] >= min_prob).sum().item()
        for gt_idx in range(1, num_gt + 1):
            gt_this = gt_array == gt_idx

            matched_prob = []
            found = False
            for pred_idx in range(1, num_pred + 1):
                pred_prob = pred_info[pred_info[:, 0] == pred_idx, 4][0]
                pred_center = pred_info[pred_info[:, 0] == pred_idx, 1:4][0]
                pred_center = slabel.TransformPhysicalPointToIndex(pred_center.tolist())[::-1]
                if pred_prob < min_prob:
                    continue
                intersection = gt_this[pred_center[0], pred_center[1], pred_center[2]]

                if intersection:
                    found = True
                    if gt_idx not in matched_gt:
                        matched_gt.append(gt_idx)
                    matched_prob.append(pred_prob)
                    if pred_idx in unmatched_pred:
                        unmatched_pred.remove(pred_idx)
                    else:  # double detection
                        doubleDetection += 1

            if found == True:
                # append the sample with the highest probability for the FROC analysis
                maxProb = np.max(matched_prob).item()
                FROCGTList.append(1.0)
                FROCProbList.append(float(maxProb))
                excludeList.append(False)
                candTPs += 1
            else:
                candFNs += 1
                FROCGTList.append(1.0)
                FROCProbList.append(minProbValue)
                excludeList.append(True)
        FPs = 0
        for unmatched_pred_idx in unmatched_pred:
            if pred_info[pred_info[:, 0] == unmatched_pred_idx, 4][0] < min_prob:
                continue
            candFPs += 1
            FPs += 1
            # remember the FPs
            FROCGTList.append(0.0)
            FROCProbList.append(pred_info[pred_info[:, 0] == unmatched_pred_idx, 4].item())
            excludeList.append(False)

        print(gt_path, 'TP: {},FN: {},FP: {}'.format(len(matched_gt), num_gt - len(matched_gt), FPs))
        del slabel, spred, gt_array, pred_array

    FPP = candFPs / patient_num
    print('{} patients, TP = {} | FN = {} | FP = {} | Pos = {}'.format(patient_num, candTPs, candFNs, candFPs, candPs))
    fps, sens, thresholds = computeFROC(FROCGTList, FROCProbList, patient_num, excludeList)
    plot_FROC(fps, sens, thresholds, key_pt=[0, 0.5, 1, 2, 4, 8], title=title)
    monitor_list = ['0.5', '1', '2', '3', '4', '5', '6']
    fp = {}
    for monitor in monitor_list:
        fp[monitor] = sens[(fps < float(monitor)).sum() - 1]
        print('{}fp {:.3f} th={:.3f}'.format(monitor, fp[monitor], thresholds[(fps < float(monitor)).sum() - 1]))
    froc_score = np.mean([fp[monitor] for monitor in monitor_list])
    print('FROC score:{:.3f}'.format(froc_score))

    info = []
    for fp, sens in zip(fps, sens):
        info.append([fp, sens])
    # with open(os.path.join('./froc', 'cad6_rnet3sv_tver_sw7525_sth50_5mm' + '.csv'), 'w', newline='') as f:
    #     ff = csv.writer(f)
    #     ff.writerow(['FPs', 'Recall'])
    #     ff.writerows(info)


from optparse import OptionParser

if __name__ == '__main__':
    val_split = np.array(pd.read_csv('./PEData/test_split_cad.csv'))[:, 0].tolist()

    gt_root = './PEData/CAD_PE_data/label'
    pred_root = './pred_itk/aanet_crop2_sth50'

    gt_list = [os.path.join(gt_root, file) for file in os.listdir(gt_root) if
               file.replace('RefStd.nrrd', '.nii.gz') in val_split]
    pred_list = [os.path.join(pred_root, file.replace('RefStd.nrrd', '.nii.gz')) for file in os.listdir(gt_root) if
                 file.replace('RefStd.nrrd', '.nii.gz') in val_split]
    pred_info_list = [os.path.join(pred_root, file.replace('RefStd.nrrd', '.csv')) for file in os.listdir(gt_root) if
                      file.replace('RefStd.nrrd', '.nii.gz') in val_split]
    gt_list.sort()
    pred_list.sort()
    pred_info_list.sort()
    fracCADEvaluation(gt_list, pred_list, pred_info_list, title=os.path.basename(pred_root), eps=2.)
