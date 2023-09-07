# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from skimage import measure, morphology
import matplotlib.pyplot as plt


def show_box3(volume, point, num=3):
    plt.figure(figsize=(12, 5))
    point = np.round(point).astype('uint16')
    index = range(point[0] - 1, point[0] + 2)

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.imshow(volume[index[i]], cmap='gray', vmin=0, vmax=1)
        if i == 1:
            plt.scatter(point[2], point[1], marker='.', s=4, c='r')
    plt.show()


def show_box2(volume, point, skip=1, width=50):
    plt.figure(figsize=(12, 5))
    point = np.round(point).astype('uint16')
    z = range(point[0] - skip, point[0] + skip + 1)
    w = width // 2
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.imshow(volume[z[i], point[1] - w:point[1] + w, point[2] - w:point[2] + w], cmap='gray', vmin=0, vmax=1)
        if i == 1:
            plt.scatter(w, w, marker='.', s=4, c='r')
    plt.show()


def show_box1(volume, point, num=1):
    plt.figure()
    point = np.round(point).astype('uint16')
    plt.imshow(volume[point[0]], cmap='gray')
    plt.scatter(point[2], point[1], marker='.', s=8, c='r')
    plt.show()


def fill_crop(img, min_idx, max_idx):
    '''
    Fills `crop` with values from `img` at `pos`,
    while accounting for the crop being off the edge of `img`.
    *Note:* negative values in `pos` are interpreted as-is, not as "from the end".
    '''
    crop = np.zeros(np.array(max_idx, dtype='int16') - np.array(min_idx, dtype='int16'), dtype=img.dtype)
    img_shape, start, crop_shape = np.array(img.shape), np.array(min_idx, dtype='int16'), np.array(crop.shape),
    end = start + crop_shape
    # Calculate crop slice positions
    crop_low = np.clip(0 - start, a_min=0, a_max=crop_shape)
    crop_high = crop_shape - np.clip(end - img_shape, a_min=0, a_max=crop_shape)
    crop_slices = (slice(low, high) for low, high in zip(crop_low, crop_high))
    # Calculate img slice positions
    pos = np.clip(start, a_min=0, a_max=img_shape)
    end = np.clip(end, a_min=0, a_max=img_shape)
    img_slices = (slice(low, high) for low, high in zip(pos, end))
    crop[tuple(crop_slices)] = img[tuple(img_slices)]
    return crop


def crop_ND_with_center_padding(volume, center_idx, output_size):
    """
    crop/extract a subregion form an nd image.
    """
    dim = len(volume.shape)
    min_idx = np.array(center_idx) - (np.array(output_size) / 2).astype('int64')
    max_idx = min_idx + np.array(output_size).astype('int64')

    assert (dim >= 2 and dim <= 5)
    assert (max_idx[0] - min_idx[0] <= volume.shape[0])
    output = fill_crop(volume, min_idx, max_idx)

    return output


def crop_ND_with_center_coord(coord, center_idx, output_size):
    """
    crop/extract a subregion form an nd image.
    """
    dim = len(center_idx)
    min_idx = np.array(center_idx) - (np.array(output_size) / 2).astype('int64')
    max_idx = min_idx + np.array(output_size).astype('int64')

    coord = np.array([loc for loc in coord if ((loc < max_idx).all() and (loc >= min_idx).all())]).reshape([-1, 3])
    coord = coord - min_idx
    return coord


def get_ND_bounding_box(volume, margin=None):
    """
    get the bounding box of nonzero region in an ND volume
    """
    input_shape = volume.shape
    if (margin is None):
        margin = [0] * len(input_shape)
    assert (len(input_shape) == len(margin))
    indxes = np.nonzero(volume)
    idx_min = []
    idx_max = []
    for i in range(len(input_shape)):
        idx_min.append(int(indxes[i].min()))
        idx_max.append(int(indxes[i].max()) + 1)

    for i in range(len(input_shape)):
        idx_min[i] = max(idx_min[i] - margin[i], 0)
        idx_max[i] = min(idx_max[i] + margin[i], input_shape[i])
    return idx_min, idx_max


def crop_ND_volume_with_bounding_box(volume, min_idx, max_idx):
    """
    crop/extract a subregion form an nd image.
    """
    dim = len(volume.shape)
    assert (dim >= 2 and dim <= 5)
    assert (max_idx[0] - min_idx[0] <= volume.shape[0])
    if (dim == 2):
        output = volume[np.ix_(range(min_idx[0], max_idx[0]),
                               range(min_idx[1], max_idx[1]))]
    elif (dim == 3):
        output = volume[np.ix_(range(min_idx[0], max_idx[0]),
                               range(min_idx[1], max_idx[1]),
                               range(min_idx[2], max_idx[2]))]
    elif (dim == 4):
        output = volume[np.ix_(range(min_idx[0], max_idx[0]),
                               range(min_idx[1], max_idx[1]),
                               range(min_idx[2], max_idx[2]),
                               range(min_idx[3], max_idx[3]))]
    elif (dim == 5):
        output = volume[np.ix_(range(min_idx[0], max_idx[0]),
                               range(min_idx[1], max_idx[1]),
                               range(min_idx[2], max_idx[2]),
                               range(min_idx[3], max_idx[3]),
                               range(min_idx[4], max_idx[4]))]
    else:
        raise ValueError("the dimension number shoud be 2 to 5")
    return output


def crop_ND_volume_with_center(volume, center_idx, output_size):
    """
    crop/extract a subregion form an nd image.
    """
    dim = len(volume.shape)
    min_idx = np.array(center_idx) - (np.array(output_size) / 2).astype('int64')
    min_idx = np.clip(min_idx, a_max=np.array(volume.shape) - 1, a_min=0)
    max_idx = min_idx + np.array(output_size).astype('int64')
    max_idx = np.clip(max_idx, a_max=np.array(volume.shape) - 1, a_min=0)
    min_idx = max_idx - np.array(output_size).astype('int64')
    assert (dim >= 2 and dim <= 5)
    assert (max_idx[0] - min_idx[0] <= volume.shape[0])
    if (dim == 2):
        output = volume[np.ix_(range(min_idx[0], max_idx[0]),
                               range(min_idx[1], max_idx[1]))]
    elif (dim == 3):
        output = volume[np.ix_(range(min_idx[0], max_idx[0]),
                               range(min_idx[1], max_idx[1]),
                               range(min_idx[2], max_idx[2]))]
    elif (dim == 4):
        output = volume[np.ix_(range(min_idx[0], max_idx[0]),
                               range(min_idx[1], max_idx[1]),
                               range(min_idx[2], max_idx[2]),
                               range(min_idx[3], max_idx[3]))]
    elif (dim == 5):
        output = volume[np.ix_(range(min_idx[0], max_idx[0]),
                               range(min_idx[1], max_idx[1]),
                               range(min_idx[2], max_idx[2]),
                               range(min_idx[3], max_idx[3]),
                               range(min_idx[4], max_idx[4]))]
    else:
        raise ValueError("the dimension number shoud be 2 to 5")
    if len(output.shape) != 3:
        raise ValueError("dimension")
    return output


def set_ND_volume_roi_with_bounding_box_range(volume, bb_min, bb_max, sub_volume, addition=True):
    """
    set a subregion to an nd image. if addition is True, the original volume is added by the subregion.
    """
    dim = len(bb_min)
    out = volume
    if (dim == 2):
        if (addition):
            out[np.ix_(range(bb_min[0], bb_max[0]),
                       range(bb_min[1], bb_max[1]))] += sub_volume
        else:
            out[np.ix_(range(bb_min[0], bb_max[0]),
                       range(bb_min[1], bb_max[1]))] = sub_volume
    elif (dim == 3):
        if (addition):
            out[np.ix_(range(bb_min[0], bb_max[0]),
                       range(bb_min[1], bb_max[1]),
                       range(bb_min[2], bb_max[2]))] += sub_volume
        else:
            out[np.ix_(range(bb_min[0], bb_max[0]),
                       range(bb_min[1], bb_max[1]),
                       range(bb_min[2], bb_max[2]))] = sub_volume
    elif (dim == 4):
        if (addition):
            out[np.ix_(range(bb_min[0], bb_max[0]),
                       range(bb_min[1], bb_max[1]),
                       range(bb_min[2], bb_max[2]),
                       range(bb_min[3], bb_max[3]))] += sub_volume
        else:
            out[np.ix_(range(bb_min[0], bb_max[0]),
                       range(bb_min[1], bb_max[1]),
                       range(bb_min[2], bb_max[2]),
                       range(bb_min[3], bb_max[3]))] = sub_volume
    elif (dim == 5):
        if (addition):
            out[np.ix_(range(bb_min[0], bb_max[0]),
                       range(bb_min[1], bb_max[1]),
                       range(bb_min[2], bb_max[2]),
                       range(bb_min[3], bb_max[3]),
                       range(bb_min[4], bb_max[4]))] += sub_volume
        else:
            out[np.ix_(range(bb_min[0], bb_max[0]),
                       range(bb_min[1], bb_max[1]),
                       range(bb_min[2], bb_max[2]),
                       range(bb_min[3], bb_max[3]),
                       range(bb_min[4], bb_max[4]))] = sub_volume
    else:
        raise ValueError("array dimension should be 2 to 5")
    return out


def get_largest_component(image):
    """
    get the largest component from 2D or 3D binary image
    image: nd array
    """
    dim = len(image.shape)
    if (image.sum() == 0):
        print('the largest component is null')
        return image
    if (dim == 2):
        s = ndimage.generate_binary_structure(2, 1)
    elif (dim == 3):
        s = ndimage.generate_binary_structure(3, 1)
    else:
        raise ValueError("the dimension number should be 2 or 3")
    labeled_array, numpatches = ndimage.label(image, s)
    sizes = ndimage.sum(image, labeled_array, range(1, numpatches + 1))
    max_label = np.where(sizes == sizes.max())[0] + 1
    output = np.asarray(labeled_array == max_label, np.uint8)
    return output


def get_euclidean_distance(image, dim=3, spacing=[1.0, 1.0, 1.0]):
    """
    get euclidean distance transform of 2D or 3D binary images
    the output distance map is unsigned
    """
    img_shape = image.shape
    input_dim = len(img_shape)
    if (input_dim != 3):
        raise ValueError("Not implemented for {0:}D image".format(input_dim))
    if (dim == 2):
        raise ValueError("Not implemented for {0:}D image".format(input_dim))
        # dis_map = np.ones_like(image, np.float32)
        # for d in range(img_shape[0]):
        #     if(image[d].sum() > 0):
        #         dis_d = ndimage.morphology.distance_transform_edt(image[d])
        #         dis_map[d] = dis_d/dis_d.max()
    elif (dim == 3):
        fg_dis_map = ndimage.morphology.distance_transform_edt(image > 0.5)
        bg_dis_map = ndimage.morphology.distance_transform_edt(image <= 0.5)
        dis_map = bg_dis_map - fg_dis_map
    else:
        raise ValueError("Not implemented for {0:}D distance".format(dim))
    return dis_map


def convert_label(label, source_list, target_list):
    """
    convert a label map based a source list and a target list of labels
    label: nd array
    source_list: a list of labels that will be converted, e.g. [0, 1, 2, 4]
    target_list: a list of target labels, e.g. [0, 1, 2, 3]
    """
    assert (len(source_list) == len(target_list))
    label_converted = np.zeros_like(label)
    for i in range(len(source_list)):
        label_temp = np.asarray(label == source_list[i], label.dtype)
        label_temp = label_temp * target_list[i]
        label_converted = label_converted + label_temp
    return label_converted


def resample_sitk_image_to_given_spacing(image, spacing, order):
    """
    image: an sitk image object
    spacing: 3D tuple / list for spacing along x, y, z direction
    order: order for interpolation
    """
    spacing0 = image.GetSpacing()
    data = sitk.GetArrayFromImage(image)
    zoom = [spacing0[i] / spacing[i] for i in range(3)]
    zoom = [zoom[2], zoom[0], zoom[1]]
    data = ndimage.interpolation.zoom(data, zoom, order=order)
    out_img = sitk.GetImageFromArray(data)
    out_img.SetSpacing(spacing)
    out_img.SetDirection(image.GetDirection())
    return out_img


def frac_mask_to_coord(mask):
    mask = mask.astype('bool')
    mask = morphology.remove_small_objects(mask, min_size=4, connectivity=3)
    label = measure.label(mask, connectivity=3)
    center = [measure.centroid(label == i) for i in range(1, label.max() + 1)]

    return center
