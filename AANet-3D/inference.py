import dataIO
import torch
import torch.nn as nn
import os
import torch.distributed
import numpy as np
import matplotlib.pyplot as plt
from models import AANet
import SimpleITK as sitk
from scipy.ndimage.morphology import binary_dilation, binary_closing
from skimage.morphology import remove_small_objects

import pandas as pd
import csv
import time
import model_utils
from utils import utils_inference, model_io
import skimage.measure

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False


def generate_binary_structure(size, distance):
    shift_z = (np.arange(0, size)) - size // 2
    shift_y = (np.arange(0, size)) - size // 2
    shift_x = (np.arange(0, size)) - size // 2
    shift_z, shift_y, shift_x = np.meshgrid(shift_z, shift_y, shift_x, indexing='ij')
    shifts = np.stack((shift_z, shift_y, shift_x), axis=-1)
    shifts = np.linalg.norm(shifts, axis=-1)
    return shifts <= distance


def calc_dst_3d(a, b):
    dist = torch.norm(torch.unsqueeze(b, dim=0) - torch.unsqueeze(a, dim=1), p=2, dim=-1)

    return dist


def nms_dst_3d(coords, scores, min_dst=3, score_thresh=0.1, top_k=100):
    coords = coords.float()
    _, idx = scores.sort(descending=True)
    keep = []
    while idx.size(0) > 0:
        i = idx[0]

        if len(keep) == top_k or (scores[i] < score_thresh):
            break
        keep.append(i)
        dst = calc_dst_3d(coords[idx], coords[i:i + 1])[:, 0]

        remained = dst >= min_dst
        idx = idx[remained]

    return torch.tensor(keep)


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


if 'lustre' in os.getcwd():
    print('On Lustre')
    batch_size = 16

else:
    print('On Local')
    batch_size = 4
DEVICE = 'cuda'

from optparse import OptionParser


def test_model(model, split_path, save_name, seg_thresh=0.5, hu_low=-100, hu_high=500):
    ct_root = './PEData/CAD_PE_data/image'
    mask_root = './PEData/CAD_PE_data/vessel'  # lung mask

    val_split = np.array(pd.read_csv(split_path))[:, 0].tolist()

    save_path = './pred_itk/' + save_name
    save_path_r1d2 = './pred_itk_r1d2/' + save_name

    if os.path.exists(save_path):
        print('dir exists')
    else:
        os.makedirs(save_path)
    if os.path.exists(save_path_r1d2):
        print('dir exists')
    else:
        os.makedirs(save_path_r1d2)

    ct_list = [os.path.join(ct_root, file) for file in os.listdir(ct_root) if
               file.replace('.nrrd', '.nii.gz') in val_split]
    mask_list = [os.path.join(mask_root, file.replace('.nrrd', '.nii.gz')) for file in os.listdir(ct_root) if
                 file.replace('.nrrd', '.nii.gz') in val_split]
    ct_list.sort()
    mask_list.sort()

    crop_size = [128, 128, 128]

    crop_fn_test = dataIO.crop_itk.SlideCrop(crop_size=crop_size, overlap=32)
    coord_generator = model_utils.Panchors3D(down_levels=[1], drift=0)
    for ct_path, mask_path in zip(ct_list, mask_list):
        file_name = os.path.basename(ct_path).replace('.nrrd', '.nii.gz')
        # if file_name != '034.nii.gz':
        #     continue
        print(file_name)
        simg = sitk.ReadImage(ct_path)
        smask = sitk.ReadImage(mask_path)

        # redirection
        if np.abs(
                np.array(simg.GetDirection()) - np.array((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))).sum() > 1:
            print(file_name, 'redirection')
            r_simg = redirection_simg(simg)
            r_smask = redirection_simg(smask)
        else:
            r_simg = simg
            r_smask = smask

        spacing = (1., 1., 1.)
        rs_simg = resample_simg(r_simg, spacing, interp=sitk.sitkLinear)
        rs_smask = resample_simg(r_smask, spacing, interp=sitk.sitkNearestNeighbor)

        np_image = sitk.GetArrayFromImage(rs_simg)
        np_mask = sitk.GetArrayFromImage(rs_smask).astype(np.uint8)

        np_image = np.clip(np_image, a_min=hu_low, a_max=hu_high)
        np_image = np_image - hu_low
        np_image = np_image.astype(np.float32) / (hu_high - hu_low)

        roi_bbox = skimage.measure.regionprops(np_mask.astype(int))[0].bbox

        "Prediction Begin"
        sample = {}
        sample['image'] = np_image
        sample['bbox'] = [roi_bbox[0], roi_bbox[1], roi_bbox[2], roi_bbox[3], roi_bbox[4], roi_bbox[5]]

        sample = crop_fn_test(sample)
        image = sample['image']
        origin_point = torch.tensor(sample['origin_point']).float().to(DEVICE)
        b, c, d, h, w = image.shape
        # continue

        all_seg = []
        all_coord = []
        all_seg_weights = []

        for bn in range(int(np.ceil(b / batch_size))):
            with torch.no_grad():
                image_batch = torch.tensor(image[bn * batch_size:(bn + 1) * batch_size]).to(DEVICE)
                result_dict = model.forward(image_batch)
                seg = result_dict['segs']
                coord = coord_generator(image_batch)
                origin = origin_point[bn * batch_size:(bn + 1) * batch_size].unsqueeze(dim=1)

                seg_weights = model_utils.panchor_weight(image.shape[2:], coord, side=32)
                coord[:, :, :3] = coord[:, :, :3] + origin

                seg = seg[:, 0].view(seg.shape[0], d, h, w, 1)
                seg_weights = seg_weights.view(seg.shape[0], d, h, w, 1)

                all_seg.append(seg)
                all_seg_weights.append(seg_weights)
                all_coord.append(coord)

        all_seg = torch.cat(all_seg, dim=0)
        all_seg_weights = torch.cat(all_seg_weights, dim=0)
        all_coord = torch.cat(all_coord, dim=0)

        all_seg = model_utils.ensemble_seg(all_seg, all_coord, all_seg_weights)
        min_origin_point = torch.min(origin_point.reshape(-1, 3), dim=0)[0].cpu()

        all_seg = all_seg.cpu().numpy()[:, :, :, 0]
        final_seg = np.zeros_like(np_image)
        min_origin_point = min_origin_point.int()
        final_seg[min_origin_point[0]:min_origin_point[0] + all_seg.shape[0],
        min_origin_point[1]:min_origin_point[1] + all_seg.shape[1],
        min_origin_point[2]:min_origin_point[2] + all_seg.shape[2]] = all_seg

        binary_seg = final_seg > seg_thresh
        # binary_seg = remove_small_objects(binary_seg, 3, connectivity=3)
        struct = generate_binary_structure(3, 3)
        # binary_seg = binary_closing(binary_seg, structure=struct, iterations=1)

        label_seg = skimage.measure.label(binary_seg)
        regions = skimage.measure.regionprops(label_seg, intensity_image=final_seg)
        seg_boxes = np.array([region.bbox for region in regions])
        seg_prob = np.array([region.max_intensity for region in regions])
        seg_id = np.array([region.label for region in regions])
        seg_center = np.array([region.centroid for region in regions])
        seg_mcenter = np.array(
            [np.array(np.unravel_index(region.intensity_image.argmax(), region.intensity_image.shape)) \
             + np.array(region.bbox[:3]) for region in regions])

        if seg_boxes.shape[0] == 0:
            info = []
            with open(os.path.join(save_path, file_name.replace('.nii.gz', '.csv')), 'w', newline='') as f:
                ff = csv.writer(f)
                ff.writerow(['id', 'x', 'y', 'z', 'prob'])
                ff.writerows(info)
            sseg = sitk.GetImageFromArray(label_seg)
            sseg.SetSpacing(rs_smask.GetSpacing())
            sseg.SetOrigin(rs_smask.GetOrigin())
            sseg.SetDirection(rs_smask.GetDirection())
            sseg = sitk.Resample(sseg, r_smask, sitk.Transform(3, sitk.sitkIdentity), sitk.sitkNearestNeighbor)
            sseg = redirection_simg(sseg, smask.GetDirection())
            sitk.WriteImage(sseg, os.path.join(save_path, file_name))
            with open(os.path.join(save_path_r1d2, file_name.replace('.nii.gz', '.csv')), 'w', newline='') as f:
                ff = csv.writer(f)
                ff.writerow(['id', 'x', 'y', 'z', 'prob'])
                ff.writerows(info)
            sitk.WriteImage(sseg, os.path.join(save_path_r1d2, file_name))
            continue

        keep = np.array(nms_dst_3d(torch.tensor(seg_mcenter), torch.tensor(seg_prob), min_dst=10))
        seg_mcenter = seg_mcenter[keep]
        seg_prob = seg_prob[keep]
        seg_id = seg_id[keep]
        centers_itk = np.array([rs_simg.TransformContinuousIndexToPhysicalPoint((seg_mcenter[i][::-1]).tolist())
                                for i in range(seg_mcenter.shape[0])])

        label_seg2 = np.zeros_like(label_seg, dtype='uint16')
        seg_id2 = np.arange(1, seg_id.shape[0] + 1)
        for i in range(seg_id.shape[0]):
            label_seg2[label_seg == seg_id[i]] = seg_id2[i]

        sseg = sitk.GetImageFromArray(label_seg2)
        sseg.SetSpacing(rs_smask.GetSpacing())
        sseg.SetOrigin(rs_smask.GetOrigin())
        sseg.SetDirection(rs_smask.GetDirection())
        sseg = sitk.Resample(sseg, r_smask, sitk.Transform(3, sitk.sitkIdentity), sitk.sitkNearestNeighbor)
        sseg = redirection_simg(sseg, smask.GetDirection())
        sitk.WriteImage(sseg, os.path.join(save_path, file_name))
        if seg_boxes.shape[0] > 0:
            info = [[int(sid), center[0], center[1], center[2], prob] for sid, center, prob in
                    zip(seg_id2, centers_itk, seg_prob)]
        else:
            info = []
        with open(os.path.join(save_path, file_name.replace('.nii.gz', '.csv')), 'w', newline='') as f:
            ff = csv.writer(f)
            ff.writerow(['id', 'x', 'y', 'z', 'prob'])
            ff.writerows(info)
        #


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('--name',
                      type='str',
                      dest='name',
                      default='aanet_sth50',
                      help='unique name of this training run')
    parser.add_option('--model',
                      type='str',
                      dest='model',
                      default='aanet',
                      help='model name of this training run')
    parser.add_option('--gpu',
                      type='str',
                      dest='gpu',
                      default='0,2',
                      help='gpu id')
    parser.add_option('--thresh',
                      type='float',
                      dest='thresh',
                      default=0.5,
                      help='thresh')
    (options, args) = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = options.gpu
    model_path = './save_models/' + options.model + '.pth'
    model = AANet.Net(n_filters=[64, 96, 128, 160], en_blocks=[2, 3, 3, 3], de_blocks=[1, 2, 2],
                      stem_filters=16, aspp_filters=32)
    print(model_path)
    model.load_state_dict(torch.load(model_path), strict=True)
    model = nn.parallel.DataParallel(model)
    model.to(DEVICE)
    model.eval()
    test_model(model, './PEData/test_split_cad.csv', options.name, seg_thresh=options.thresh,
               hu_low=-100, hu_high=500)
