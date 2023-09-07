from scipy import ndimage
import numpy as np
import SimpleITK as sitk
from skimage.measure import regionprops
import re
import os
import pandas as pd
import dataIO

label_code_dict = {
    0: "Background",
    1: "Displaced",
    2: "Nondisplaced",
    3: "Buckle",
    4: "Segmental",
    -1: "Ignore"
}


def solve_affine(x, y):
    x = np.transpose(x)
    y = np.transpose(y)
    # add ones on the bottom of x and y
    x = np.vstack((x, [1, 1, 1, 1]))
    y = np.vstack((y, [1, 1, 1, 1]))
    # solve for A2
    A2 = y * np.linalg.inv(x)
    # return function that takes input x and transforms it
    # don't need to return the 4th row as it is

    return lambda x: (A2 * np.vstack((np.array(x).reshape(3, 1), 1)))[0:3, :]


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


def convert_nifty(image, label, mask, info=None, pid=None, margin=0):
    # redirection
    if np.abs(
            np.array(mask.GetDirection()) - np.array((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))).sum() > 1:
        image = redirection_simg(image)
        label = redirection_simg(label)
        mask = redirection_simg(mask)

    # resample to 1mm and save
    image = resample_simg(image, interp=sitk.sitkLinear)
    label = resample_simg(label, interp=sitk.sitkNearestNeighbor)
    mask = resample_simg(mask, interp=sitk.sitkNearestNeighbor)

    # to numpy
    np_image = sitk.GetArrayFromImage(image).astype('float32')
    np_label = sitk.GetArrayFromImage(label).astype('uint8')
    np_mask = sitk.GetArrayFromImage(mask).astype('uint8')

    low_th = -1000
    high_th = 500
    np_image = np.clip(np_image, a_min=low_th, a_max=high_th)
    np_image = np_image - low_th
    np_image = np_image / (high_th - low_th)
    np_image = np_image.astype(np.float32)

    if info is not None:
        gt_info = info.loc[info.public_id == pid].reset_index(drop=True)
    else:
        gt_info = []

    rib_arg = np.array(np.where(np_mask > 0))
    rib_min = rib_arg.min(axis=1)
    rib_max = rib_arg.max(axis=1)

    cage_start = np.clip(rib_min - rib_cage, a_min=0, a_max=None)
    cage_end = np.clip(rib_max + rib_cage, a_min=None, a_max=np.array(np_image.shape))

    np_image = np_image[cage_start[0]:cage_end[0], cage_start[1]:cage_end[1], cage_start[2]:cage_end[2]]
    np_label = np_label[cage_start[0]:cage_end[0], cage_start[1]:cage_end[1], cage_start[2]:cage_end[2]]
    np_mask = np_mask[cage_start[0]:cage_end[0], cage_start[1]:cage_end[1], cage_start[2]:cage_end[2]]

    return np_image, np_label, np_mask, gt_info


if __name__ == "__main__":
    import torchvision

    image_root = '/data/guojia/RibFrac_miccai2020/train'
    label_root = '/data/guojia/RibFrac_miccai2020/train_label'
    mask_root = '/data/guojia/RibFrac_miccai2020/train_mask24'
    info_path = '/data/guojia/RibFrac_miccai2020/ribfrac-train-info.csv'

    crop_size = [96, 96, 96]
    transform_list = [
        transform.RandomFlip(flip_depth=False, flip_height=False, flip_width=True),
        transform.Pad(output_size=crop_size),
        transform.RandomCrop(output_size=crop_size),
        transform.RandomGamma(gamma_range=1.4, p=0.3),
        transform.RandomBlur(sigma_range=(0.4, 0.7), p=0.15),
        transform.RandomNoise(gamma_range=(1e-4, 5e-4), p=0.3),
        transform.LabelToAnnot(blank_side=2)]
    train_transform = torchvision.transforms.Compose(transform_list)

    crop_fn = dataIO.crop4.RibCrop(output_size=[96, 96, 96], pos_ratio=0.7,
                                   rand_trans=[10, 5, 5], rand_rot=[10, 10, 10], rand_space=[0.95, 1.1],
                                   sample_num=2)
    dataset = dataIO.dataset.RibDatasetNifty(image_root, label_root, mask_root, info_path,
                                             crop_fn=crop_fn, transform_post=train_transform)

    for i in range(3, 8):
        samples = dataset[i]
