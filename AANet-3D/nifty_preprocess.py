import SimpleITK as sitk
from scipy.ndimage.morphology import binary_closing, generate_binary_structure
from scipy import ndimage
import pandas as pd
import csv
import time
from skimage.measure import regionprops, label
from skimage.morphology import disk
import numpy as np
import os
import matplotlib.pyplot as plt
from utils.itkprocess import skeletonSegment


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


ct_roots = ['./PEData/CAD_PE_data/images']
label_roots = ['./PEData/CAD_PE_data/label']
mask_roots = ['./PEData/CAD_PE_data/vessel']
vessel_roots = ['./PEData/CAD_PE_data/vessel']

save_ct_root = './PEData/processed_itk/image'
save_label_root = './PEData/processed_itk/label'
save_vessel_root = './PEData/processed_itk/vessel'

for ct_root, label_root, mask_root, vessel_root in zip(ct_roots, label_roots, mask_roots, vessel_roots):
    files = os.listdir(ct_root)

    for file in files:

        ct_path = os.path.join(ct_root, file)
        label_path = os.path.join(label_root, file.replace('.nrrd', 'RefStd.nrrd'))
        mask_path = os.path.join(mask_root, file.replace('.nrrd', '.nii.gz').replace('.nii.gz', '_lungmask.nii.gz'))
        vessel_path = os.path.join(vessel_root, file.replace('.nrrd', '.nii.gz'))

        file_name = os.path.basename(ct_path).replace('.nrrd', '.nii.gz')

        # load data
        simg = sitk.ReadImage(ct_path)
        slabel = sitk.ReadImage(label_path)
        smask = sitk.ReadImage(mask_path)
        svessel = sitk.ReadImage(vessel_path)

        print(simg.GetSpacing())

        # redirection
        if np.abs(
                np.array(simg.GetDirection()) - np.array((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))).sum() > 1:
            print(file_name, 'redirection')
            simg = redirection_simg(simg)
            slabel = redirection_simg(slabel)
            smask = redirection_simg(smask)
            svessel = redirection_simg(svessel)

        np_image = sitk.GetArrayFromImage(simg)
        np_label = sitk.GetArrayFromImage(slabel).astype(np.uint8)
        np_mask = sitk.GetArrayFromImage(smask).astype(np.uint8)
        np_vessel = sitk.GetArrayFromImage(svessel).astype(np.uint8)

        assert np_mask.shape == np_image.shape
        assert np_image.shape == np_label.shape

        # lung area
        roi_bbox = regionprops(np_mask.astype(int))[0].bbox
        np_image = np_image[roi_bbox[0]:roi_bbox[3], roi_bbox[1]:roi_bbox[4], roi_bbox[2]:roi_bbox[5]]
        np_label = np_label[roi_bbox[0]:roi_bbox[3], roi_bbox[1]:roi_bbox[4], roi_bbox[2]:roi_bbox[5]]
        np_vessel = np_vessel[roi_bbox[0]:roi_bbox[3], roi_bbox[1]:roi_bbox[4], roi_bbox[2]:roi_bbox[5]]

        # in case embolism is not contained by vessel
        np_vessel[np_label > 0] = 1

        itk_image = sitk.GetImageFromArray(np_image)
        itk_image.SetSpacing(slabel.GetSpacing())
        itk_label = sitk.GetImageFromArray(np_label)
        itk_label.SetSpacing(slabel.GetSpacing())
        itk_vessel = sitk.GetImageFromArray(np_vessel)
        itk_vessel.SetSpacing(slabel.GetSpacing())

        sitk.WriteImage(itk_image, os.path.join(save_ct_root, file_name))
        sitk.WriteImage(itk_label, os.path.join(save_label_root, file_name))
        sitk.WriteImage(itk_vessel, os.path.join(save_vessel_root, file_name))
