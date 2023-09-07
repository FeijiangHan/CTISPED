# -*- coding: utf-8 -*-
from __future__ import print_function, division
import SimpleITK as sitk
import numpy as np
from scipy import ndimage
from skimage import measure, morphology
from matplotlib import pyplot as plt
import random
import skimage.measure
from scipy.ndimage import gaussian_filter
from scipy.ndimage import distance_transform_edt


def one_hot(a, num_classes):
    return np.eye(num_classes, dtype='uint8')[a.reshape(-1)].transpose(1, 0).reshape([num_classes] + list(a.shape))


def apply_transformation_coord(coord, transform_param_list, rot_center):
    """
    apply rotation transformation to an ND image
    Args:
        image (nd array): the input nd image
        transform_param_list (list): a list of roration angle and axes
        order (int): interpolation order
    """
    for angle, axes in transform_param_list:
        # rot_center = np.random.uniform(low=np.min(coord, axis=0), high=np.max(coord, axis=0), size=3)
        org = coord - rot_center
        new = rotate_vecs_3d(org, angle, axes)
        coord = new + rot_center

    return coord


def rand_rot_coord(coord, angle_range_d, angle_range_h, angle_range_w, rot_center, p):
    transform_param_list = []

    if (angle_range_d is not None) and random.random() < p:
        angle_d = np.random.uniform(angle_range_d[0], angle_range_d[1])
        transform_param_list.append([angle_d, (-2, -1)])
    if (angle_range_h is not None) and random.random() < p:
        angle_h = np.random.uniform(angle_range_h[0], angle_range_h[1])
        transform_param_list.append([angle_h, (-3, -1)])
    if (angle_range_w is not None) and random.random() < p:
        angle_w = np.random.uniform(angle_range_w[0], angle_range_w[1])
        transform_param_list.append([angle_w, (-3, -2)])

    if len(transform_param_list) > 0:
        coord = apply_transformation_coord(coord, transform_param_list, rot_center)

    return coord


def rand_rot90_coord(coord, rot_center, p, rot_d=False, rot_h=False, rot_w=False):
    transform_param_list = []

    if rot_d and random.random() < p:
        angle_d = np.random.choice([0, 90, 180, 270])
        transform_param_list.append([angle_d, (-2, -1)])
    if rot_h and random.random() < p:
        angle_h = np.random.choice([0, 90, 180, 270])
        transform_param_list.append([angle_h, (-3, -1)])
    if rot_w and random.random() < p:
        angle_w = np.random.choice([0, 90, 180, 270])
        transform_param_list.append([angle_w, (-3, -2)])

    if len(transform_param_list) > 0:
        coord = apply_transformation_coord(coord, transform_param_list, rot_center)

    return coord


def sample_crop(itk_img, mark_matrix, spacing=[1., 1., 1.], interp1=sitk.sitkLinear):
    '''
    itk_img: image to reorient
    mark_matric: physical mark point
    '''
    spacing = spacing[::-1]
    origin, x_mark, y_mark, z_mark = np.array(mark_matrix[0]), np.array(mark_matrix[1]), np.array(
        mark_matrix[2]), np.array(mark_matrix[3])

    # centroid_world = itk_img.TransformContinuousIndexToPhysicalPoint(centroid)
    filter_resample = sitk.ResampleImageFilter()
    filter_resample.SetInterpolator(interp1)
    filter_resample.SetOutputSpacing(spacing)

    # set origin
    origin_reorient = mark_matrix[0]
    # set direction
    # !!! note: column wise
    x_base = (x_mark - origin) / np.linalg.norm(x_mark - origin)
    y_base = (y_mark - origin) / np.linalg.norm(y_mark - origin)
    z_base = (z_mark - origin) / np.linalg.norm(z_mark - origin)
    direction_reorient = np.stack([x_base, y_base, z_base]).transpose().reshape(-1).tolist()

    # set size
    x, y, z = np.linalg.norm(x_mark - origin) / spacing[0], np.linalg.norm(y_mark - origin) / spacing[
        1], np.linalg.norm(z_mark - origin) / spacing[2]
    size_reorient = (int(np.ceil(x + 0.5)), int(np.ceil(y + 0.5)), int(np.ceil(z + 0.5)))

    filter_resample.SetOutputOrigin(origin_reorient)
    filter_resample.SetOutputDirection(direction_reorient)
    filter_resample.SetSize(size_reorient)
    # filter_resample.SetSpacing([sp]*3)

    filter_resample.SetOutputPixelType(itk_img.GetPixelID())
    itk_out = filter_resample.Execute(itk_img)

    return itk_out


def resample_simg(simg, interp=sitk.sitkBSpline, spacing=[1., 1., 1.]):
    identity1 = sitk.Transform(3, sitk.sitkIdentity)
    new_spacing = spacing[::-1]

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


def shrink_matrix(matrix, side=0):
    center = matrix[1] / 2 + matrix[2] / 2 + matrix[3] / 2 - matrix[0] / 2
    x_vector = matrix[1] - matrix[0]
    y_vector = matrix[2] - matrix[0]
    z_vector = matrix[3] - matrix[0]
    x_vector_de = x_vector / np.linalg.norm(x_vector) * side
    y_vector_de = y_vector / np.linalg.norm(y_vector) * side
    z_vector_de = z_vector / np.linalg.norm(z_vector) * side

    O = matrix[0] + x_vector_de + y_vector_de + z_vector_de
    X = O + x_vector - 2 * x_vector_de
    Y = O + y_vector - 2 * y_vector_de
    Z = O + z_vector - 2 * z_vector_de
    matrix = np.array([O, X, Y, Z])
    return matrix


class SegCrop(object):
    """Randomly crop the input image (shape [C, D, H, W]
    """

    def __init__(self, crop_size,
                 rand_translation=None, rand_rotation=None, rand_rotation90=None,
                 rand_space=None, rand_flip=None, rand_transpose=None,
                 obj_crop=True, spacing=[1., 1., 1.], hu_low=-1000, hu_high=500,
                 overlap=[32, 32, 32], tp_ratio=0.7, sample_num=2, blank_side=8):
        """This is crop function with spatial augmentation for training Lesion Detection.
         It crops patches according to rib mask. It use simpleITK to fasten the augmentation.

        Arguments:
            crop_size: patch size
            rand_transation: random translation
            rand_rotation: random rotation
            rand_space: random spacing
            obj_crop: additional sampling with fracture around center
            spacing: output patch spacing, [z,y,x]
            base_spacing: spacing of the numpy image.
            overlap: overlap of sliding window
            tp_ratio: sampling rate for a patch containing at least one leision
            sample_num: patch number per CT
            blank_side:  labels within blank_side pixels near patch border is set to ignored.

        """
        self.crop_size = crop_size
        self.tp_ratio = tp_ratio
        self.sample_num = sample_num
        self.blank_side = blank_side
        self.obj_crop = obj_crop
        self.overlap = overlap
        self.spacing = spacing
        self.hu_low = hu_low
        self.hu_high = hu_high

        self.rand_translation = None if rand_translation is None else np.array(rand_translation)[::-1]
        self.rand_rotation = None if rand_rotation is None else np.array(rand_rotation)[::-1]
        self.rand_rotation90 = None if rand_rotation90 is None else np.array(rand_rotation90)[::-1]
        self.rand_space = None if rand_space is None else np.array(rand_space)[::-1]
        self.rand_flip = None if rand_flip is None else np.array(rand_flip)[::-1]
        self.rand_transpose = None if rand_transpose is None else np.array(rand_transpose)[::-1]

        assert isinstance(self.crop_size, (list, tuple))

    def __call__(self, sample):
        image_itk = sample['image_itk']
        label_itk = sample['label_itk']
        itk_spacing = label_itk.GetSpacing()[::-1]

        label = sitk.GetArrayFromImage(label_itk).astype(np.uint8)

        dlabel = skimage.measure.label(np.clip(label, a_min=0, a_max=1))
        regions = skimage.measure.regionprops(dlabel)
        obj_ctr = np.array([region.centroid for region in regions]).reshape(-1, 3)
        obj_id = np.array([region.label for region in regions])
        shape = label.shape

        space_crop_size = np.array(self.crop_size) * np.array(self.spacing)
        re_spacing = np.array(self.spacing) / np.array(itk_spacing)
        pixel_crop_size = space_crop_size * re_spacing
        overlap = np.array(self.overlap) * re_spacing

        z_stride = pixel_crop_size[0] - overlap[0]
        y_stride = pixel_crop_size[1] - overlap[1]
        x_stride = pixel_crop_size[2] - overlap[2]

        z_range = np.arange(0, shape[0] - overlap[0], z_stride)
        y_range = np.arange(0, shape[1] - overlap[1], y_stride)
        x_range = np.arange(0, shape[2] - overlap[2], x_stride)

        crop_starts = []
        for z in z_range:
            for y in y_range:
                for x in x_range:
                    crop_starts.append(np.array([z, y, x]))
        crop_starts = np.array(crop_starts)

        if self.obj_crop:
            if self.rand_translation is not None:
                obj_crop = obj_ctr + np.random.randint(low=-pixel_crop_size / 4, high=pixel_crop_size / 4, size=3)
            else:
                obj_crop = obj_ctr
            crop_starts = np.append(crop_starts, obj_crop - (pixel_crop_size / 2), axis=0)

        crop_starts = np.clip(crop_starts, a_max=np.array(shape) - pixel_crop_size, a_min=np.array([0, 0, 0]))

        obj_ctr_itk = np.array([label_itk.TransformIndexToPhysicalPoint((obj_ctr[i][::-1].astype('int64') + 1).tolist())
                                for i in range(obj_ctr.shape[0])])
        crop_starts_itk = np.array([
            label_itk.TransformIndexToPhysicalPoint((crop_starts[i][::-1].astype('int64') + 1).tolist())
            for i in range(crop_starts.shape[0])])

        tp_num = []
        matrix_crops = []
        space_crops = []
        for i in range(len(crop_starts_itk)):
            O = crop_starts_itk[i]

            if self.rand_translation is not None:
                O = O + np.random.randint(low=-self.rand_translation, high=self.rand_translation, size=3)

            Center = O + space_crop_size[::-1] / 2
            X = O + np.array([space_crop_size[2] - 1, 0, 0])
            Y = O + np.array([0, space_crop_size[1] - 1, 0])
            Z = O + np.array([0, 0, space_crop_size[0] - 1])
            matrix = np.array([O, X, Y, Z])

            if self.rand_rotation90 is not None:
                matrix = rand_rot90_coord(matrix, rot_center=Center, p=0.8, rot_d=self.rand_rotation90[0],
                                          rot_h=self.rand_rotation90[1], rot_w=self.rand_rotation90[2])

            if self.rand_rotation is not None:
                matrix = rand_rot_coord(matrix, [-self.rand_rotation[0], self.rand_rotation[0]],
                                        [-self.rand_rotation[1], self.rand_rotation[1]],
                                        [-self.rand_rotation[2], self.rand_rotation[2]], rot_center=Center, p=0.8)

            if (self.rand_space is not None) and (random.random() < 0.8):
                space = np.random.uniform(self.rand_space[0], self.rand_space[1], size=3) * np.array(self.spacing)
            else:
                space = np.array(self.spacing)

            if self.rand_flip is not None:
                if self.rand_flip[0] and (random.random() < 0.5):
                    trans = matrix[1] - matrix[0]
                    matrix[0], matrix[1] = matrix[1].copy(), matrix[0].copy()
                    matrix[2] = matrix[2] + trans
                    matrix[3] = matrix[3] + trans

                if self.rand_flip[1] and (random.random() < 0.5):
                    trans = matrix[2] - matrix[0]
                    matrix[0], matrix[2] = matrix[2].copy(), matrix[0].copy()
                    matrix[1] = matrix[1] + trans
                    matrix[3] = matrix[3] + trans

                if self.rand_flip[2] and (random.random() < 0.5):
                    trans = matrix[3] - matrix[0]
                    matrix[0], matrix[3] = matrix[3].copy(), matrix[0].copy()
                    matrix[1] = matrix[1] + trans
                    matrix[2] = matrix[2] + trans

            if self.rand_transpose is not None:
                if self.rand_transpose[0] and (random.random() < 0.5):
                    matrix[2], matrix[3] = matrix[3].copy(), matrix[2].copy()
                if self.rand_transpose[1] and (random.random() < 0.5):
                    matrix[1], matrix[3] = matrix[3].copy(), matrix[1].copy()
                if self.rand_transpose[2] and (random.random() < 0.5):
                    matrix[1], matrix[2] = matrix[2].copy(), matrix[1].copy()

            # matrix = matrix[:, ::-1]  # in itk axis
            label_itk_crop = sample_crop(label_itk, shrink_matrix(matrix, side=self.blank_side), spacing=list(space),
                                         interp1=sitk.sitkNearestNeighbor)

            obj_ctr_crop = [label_itk_crop.TransformPhysicalPointToContinuousIndex(c.tolist())[::-1] for c in
                            obj_ctr_itk]
            obj_ctr_crop = np.array(obj_ctr_crop)

            in_idx = []
            for j in range(obj_ctr_crop.shape[0]):
                if (obj_ctr_crop[j] <= np.array(label_itk_crop.GetSize()[::-1])).all() and (
                        obj_ctr_crop[j] >= np.zeros([3])).all():
                    in_idx.append(True)
                else:
                    in_idx.append(False)
            in_idx = np.array(in_idx)

            if in_idx.size > 0:
                obj_ctr_crop = obj_ctr_crop[in_idx]
            else:
                obj_ctr_crop = np.array([]).reshape(-1, 3)

            tp_num.append(obj_ctr_crop.shape[0])

            matrix_crops.append(matrix)
            space_crops.append(space)

        tp_num = np.array(tp_num)
        tp_idx = tp_num > 0
        neg_idx = tp_num == 0

        if tp_idx.sum() > 0:
            tp_pos = self.tp_ratio / tp_idx.sum()
        else:
            tp_pos = 0

        p = np.zeros(shape=tp_num.shape)
        p[tp_idx] = tp_pos
        p[neg_idx] = (1. - p.sum()) / neg_idx.sum() if neg_idx.sum() > 0 else 0
        p = p * 1 / p.sum()

        index = np.random.choice(np.arange(len(crop_starts)), size=self.sample_num, p=p)

        image_crops = []
        label_crops = []

        for i in index:
            matrix = matrix_crops[i]
            space = space_crops[i]
            image_itk_crop = sample_crop(image_itk, matrix, spacing=list(space), interp1=sitk.sitkLinear)
            image_crop = sitk.GetArrayFromImage(image_itk_crop)

            label_itk_crop = sample_crop(label_itk, matrix, spacing=list(space), interp1=sitk.sitkNearestNeighbor)
            label_crop = sitk.GetArrayFromImage(label_itk_crop)
            label_crop = label_crop > 0

            low_th = self.hu_low
            high_th = self.hu_high
            image_crop = np.clip(image_crop, a_min=low_th, a_max=high_th)
            image_crop = image_crop - low_th
            image_crop = image_crop.astype(np.float32) / (high_th - low_th)
            image_crops.append(np.expand_dims(image_crop, axis=0))
            label_crops.append(np.expand_dims(label_crop, axis=0))

        samples = []
        for i in range(len(image_crops)):
            sample = {}
            sample['image'] = image_crops[i]
            sample['label'] = label_crops[i]

            samples.append(sample)

        return samples


class SegCrop2(object):
    """Randomly crop the input image (shape [C, D, H, W]
    """

    def __init__(self, crop_size,
                 rand_translation=None, rand_rotation=None, rand_rotation90=None,
                 rand_space=None, rand_flip=None, rand_transpose=None,
                 obj_crop=True, spacing=[1., 1., 1.], hu_low=-100, hu_high=500,
                 overlap=[32, 32, 32], tp_ratio=0.7, sample_num=2, vessel_num=1, blank_side=0):
        """This is crop function with spatial augmentation for training Lesion Detection.
         It crops patches according to rib mask. It use simpleITK to fasten the augmentation.

        Arguments:
            crop_size: patch size
            rand_transation: random translation
            rand_rotation: random rotation
            rand_space: random spacing
            obj_crop: additional sampling with fracture around center
            spacing: output patch spacing, [z,y,x]
            base_spacing: spacing of the numpy image.
            overlap: overlap of sliding window
            tp_ratio: sampling rate for a patch containing at least one leision
            sample_num: patch number per CT
            blank_side:  labels within blank_side pixels near patch border is set to ignored.

        """
        self.crop_size = crop_size
        self.tp_ratio = tp_ratio
        self.sample_num = sample_num
        self.obj_crop = obj_crop
        self.overlap = overlap
        self.spacing = spacing
        self.hu_low = hu_low
        self.hu_high = hu_high
        self.vessel_num = vessel_num
        self.blank_side = blank_side

        self.rand_translation = None if rand_translation is None else np.array(rand_translation)[::-1]
        self.rand_rotation = None if rand_rotation is None else np.array(rand_rotation)[::-1]
        self.rand_rotation90 = None if rand_rotation90 is None else np.array(rand_rotation90)[::-1]
        self.rand_space = None if rand_space is None else np.array(rand_space)[::-1]
        self.rand_flip = None if rand_flip is None else np.array(rand_flip)[::-1]
        self.rand_transpose = None if rand_transpose is None else np.array(rand_transpose)[::-1]

        assert isinstance(self.crop_size, (list, tuple))

    def __call__(self, sample):
        image_itk = sample['image_itk']
        label_itk = sample['label_itk']
        vessel_itk = sample['vessel_itk']

        itk_spacing = label_itk.GetSpacing()[::-1]

        label = sitk.GetArrayFromImage(label_itk).astype(np.uint8)

        dlabel = skimage.measure.label(np.clip(label, a_min=0, a_max=1))
        dlabel = morphology.remove_small_objects(dlabel, 2)
        regions = skimage.measure.regionprops(dlabel)
        obj_ctr = np.array([region.centroid for region in regions]).reshape(-1, 3)
        obj_id = np.array([region.label for region in regions])
        shape = label.shape

        space_crop_size = np.array(self.crop_size) * np.array(self.spacing)
        re_spacing = np.array(self.spacing) / np.array(itk_spacing)
        pixel_crop_size = space_crop_size * re_spacing
        overlap = np.array(self.overlap) * re_spacing

        z_stride = pixel_crop_size[0] - overlap[0]
        y_stride = pixel_crop_size[1] - overlap[1]
        x_stride = pixel_crop_size[2] - overlap[2]

        z_range = np.arange(0, shape[0] - overlap[0], z_stride)
        y_range = np.arange(0, shape[1] - overlap[1], y_stride)
        x_range = np.arange(0, shape[2] - overlap[2], x_stride)

        crop_starts = []
        for z in z_range:
            for y in y_range:
                for x in x_range:
                    crop_starts.append(np.array([z, y, x]))
        crop_starts = np.array(crop_starts)

        if self.obj_crop:
            if self.rand_translation is not None:
                obj_crop = obj_ctr + np.random.randint(low=-pixel_crop_size / 4, high=pixel_crop_size / 4, size=3)
            else:
                obj_crop = obj_ctr
            crop_starts = np.append(crop_starts, obj_crop - (pixel_crop_size / 2), axis=0)

        crop_starts = np.clip(crop_starts, a_max=np.array(shape) - pixel_crop_size, a_min=np.array([0, 0, 0]))

        obj_ctr_itk = np.array([label_itk.TransformIndexToPhysicalPoint((obj_ctr[i][::-1].astype('int64') + 1).tolist())
                                for i in range(obj_ctr.shape[0])])
        crop_starts_itk = np.array([
            label_itk.TransformIndexToPhysicalPoint((crop_starts[i][::-1].astype('int64') + 1).tolist())
            for i in range(crop_starts.shape[0])])

        tp_num = []
        matrix_crops = []
        space_crops = []
        for i in range(len(crop_starts_itk)):
            O = crop_starts_itk[i]

            if self.rand_translation is not None:
                O = O + np.random.randint(low=-self.rand_translation, high=self.rand_translation, size=3)

            Center = O + space_crop_size[::-1] / 2
            X = O + np.array([space_crop_size[2] - 1, 0, 0])
            Y = O + np.array([0, space_crop_size[1] - 1, 0])
            Z = O + np.array([0, 0, space_crop_size[0] - 1])
            matrix = np.array([O, X, Y, Z])

            if self.rand_rotation90 is not None:
                matrix = rand_rot90_coord(matrix, rot_center=Center, p=0.8, rot_d=self.rand_rotation90[0],
                                          rot_h=self.rand_rotation90[1], rot_w=self.rand_rotation90[2])

            if self.rand_rotation is not None:
                matrix = rand_rot_coord(matrix, [-self.rand_rotation[0], self.rand_rotation[0]],
                                        [-self.rand_rotation[1], self.rand_rotation[1]],
                                        [-self.rand_rotation[2], self.rand_rotation[2]], rot_center=Center, p=0.8)

            if (self.rand_space is not None) and (random.random() < 0.8):
                space = np.random.uniform(self.rand_space[0], self.rand_space[1], size=3) * np.array(self.spacing)
            else:
                space = np.array(self.spacing)

            if self.rand_flip is not None:
                if self.rand_flip[0] and (random.random() < 0.5):
                    trans = matrix[1] - matrix[0]
                    matrix[0], matrix[1] = matrix[1].copy(), matrix[0].copy()
                    matrix[2] = matrix[2] + trans
                    matrix[3] = matrix[3] + trans

                if self.rand_flip[1] and (random.random() < 0.5):
                    trans = matrix[2] - matrix[0]
                    matrix[0], matrix[2] = matrix[2].copy(), matrix[0].copy()
                    matrix[1] = matrix[1] + trans
                    matrix[3] = matrix[3] + trans

                if self.rand_flip[2] and (random.random() < 0.5):
                    trans = matrix[3] - matrix[0]
                    matrix[0], matrix[3] = matrix[3].copy(), matrix[0].copy()
                    matrix[1] = matrix[1] + trans
                    matrix[2] = matrix[2] + trans

            if self.rand_transpose is not None:
                if self.rand_transpose[0] and (random.random() < 0.5):
                    matrix[2], matrix[3] = matrix[3].copy(), matrix[2].copy()
                if self.rand_transpose[1] and (random.random() < 0.5):
                    matrix[1], matrix[3] = matrix[3].copy(), matrix[1].copy()
                if self.rand_transpose[2] and (random.random() < 0.5):
                    matrix[1], matrix[2] = matrix[2].copy(), matrix[1].copy()

            # matrix = matrix[:, ::-1]  # in itk axis
            label_itk_crop = sample_crop(label_itk, shrink_matrix(matrix, side=self.blank_side), spacing=list(space),
                                         interp1=sitk.sitkNearestNeighbor)

            obj_ctr_crop = [label_itk_crop.TransformPhysicalPointToContinuousIndex(c.tolist())[::-1] for c in
                            obj_ctr_itk]
            obj_ctr_crop = np.array(obj_ctr_crop)

            in_idx = []
            for j in range(obj_ctr_crop.shape[0]):
                if (obj_ctr_crop[j] <= np.array(label_itk_crop.GetSize()[::-1])).all() and (
                        obj_ctr_crop[j] >= np.zeros([3])).all():
                    in_idx.append(True)
                else:
                    in_idx.append(False)
            in_idx = np.array(in_idx)

            if in_idx.size > 0:
                obj_ctr_crop = obj_ctr_crop[in_idx]
            else:
                obj_ctr_crop = np.array([]).reshape(-1, 3)

            tp_num.append(obj_ctr_crop.shape[0])

            matrix_crops.append(matrix)
            space_crops.append(space)

        tp_num = np.array(tp_num)
        tp_idx = tp_num > 0
        neg_idx = tp_num == 0

        if tp_idx.sum() > 0:
            tp_pos = self.tp_ratio / tp_idx.sum()
        else:
            tp_pos = 0

        p = np.zeros(shape=tp_num.shape)
        p[tp_idx] = tp_pos
        p[neg_idx] = (1. - p.sum()) / neg_idx.sum() if neg_idx.sum() > 0 else 0
        p = p * 1 / p.sum()

        index = np.random.choice(np.arange(len(crop_starts)), size=self.sample_num, p=p)

        image_crops = []
        label_crops = []
        vessel_crops = []
        for i in index:
            matrix = matrix_crops[i]
            space = space_crops[i]
            image_itk_crop = sample_crop(image_itk, matrix, spacing=list(space), interp1=sitk.sitkLinear)
            image_crop = sitk.GetArrayFromImage(image_itk_crop)
            label_itk_crop = sample_crop(label_itk, matrix, spacing=list(space), interp1=sitk.sitkNearestNeighbor)
            label_crop = sitk.GetArrayFromImage(label_itk_crop)
            vessel_itk_crop = sample_crop(vessel_itk, matrix, spacing=list(space), interp1=sitk.sitkNearestNeighbor)
            vessel_crop = sitk.GetArrayFromImage(vessel_itk_crop)
            vessel_crop[label_crop == 1] = 1
            vessel_crop[vessel_crop > self.vessel_num] = 0

            low_th = self.hu_low
            high_th = self.hu_high
            image_crop = np.clip(image_crop, a_min=low_th, a_max=high_th)
            image_crop = image_crop - low_th
            image_crop = image_crop.astype(np.float32) / (high_th - low_th)
            label_crop = np.clip(label_crop, a_min=0, a_max=1)

            image_crops.append(np.expand_dims(image_crop, axis=0))
            label_crops.append(np.expand_dims(label_crop, axis=0))
            vessel_crops.append(np.expand_dims(vessel_crop, axis=0))

        samples = []
        for i in range(len(image_crops)):
            sample = {}
            sample['image'] = image_crops[i]
            sample['label'] = label_crops[i]
            sample['mask'] = vessel_crops[i]

            samples.append(sample)

        return samples


def rotate_vecs_3d(vec, angle, axis):
    rad = np.deg2rad(angle)
    rotated_vec = vec.copy()
    rotated_vec[::, axis[0]] = vec[::, axis[0]] * np.cos(rad) - vec[::, axis[1]] * np.sin(rad)
    rotated_vec[::, axis[1]] = vec[::, axis[0]] * np.sin(rad) + vec[::, axis[1]] * np.cos(rad)
    return rotated_vec


class SlideCrop(object):
    """Randomly crop the input image (shape [C, D, H, W] or [C, H, W])
    """

    def __init__(self, crop_size, overlap=24, divider=4):
        """

        """
        self.crop_size = crop_size
        self.overlap = overlap
        self.divider = divider

    def __call__(self, sample):
        image = sample['image'].astype('float32')

        bbox = sample['bbox']

        z_stride = self.crop_size[0] - self.overlap
        y_stride = self.crop_size[1] - self.overlap
        x_stride = self.crop_size[2] - self.overlap

        z_range = np.arange(bbox[0], bbox[3] - self.overlap, z_stride)
        y_range = np.arange(bbox[1], bbox[4] - self.overlap, y_stride)
        x_range = np.arange(bbox[2], bbox[5] - self.overlap, x_stride)

        z_range = np.clip(z_range, a_max=bbox[3] - self.crop_size[0], a_min=None)
        y_range = np.clip(y_range, a_max=bbox[4] - self.crop_size[1], a_min=None)
        x_range = np.clip(x_range, a_max=bbox[5] - self.crop_size[2], a_min=None)

        crop_starts = []
        for z in z_range:
            for y in y_range:
                for x in x_range:
                    crop_starts.append(np.array([z // self.divider * self.divider,
                                                 y // self.divider * self.divider,
                                                 x // self.divider * self.divider]))
        CT_crops = []
        origin_point = []
        for i in range(len(crop_starts)):
            image_crop = crop_ND_with_start_padding(image, crop_starts[i], self.crop_size)

            CT_crops.append(np.expand_dims(image_crop, axis=0))
            origin_point.append(np.array(crop_starts[i]).astype('int32'))

        CT_crops = np.stack(CT_crops)
        origin_point = np.stack(origin_point)

        sample = {}
        sample['image'] = CT_crops
        sample['origin_point'] = origin_point
        return sample


def displacement(A, B):
    disp = []
    for i in range(A.shape[0]):
        adisp = []
        for j in range(B.shape[0]):
            adisp.append(B[j] - A[i])
        disp.append(adisp)
    disp = np.array(disp)
    return disp


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
    # assert (max_idx[0] - min_idx[0] <= volume.shape[0])
    output = fill_crop(volume, min_idx, max_idx)

    return output


def crop_ND_with_start_padding(volume, start_idx, output_size):
    """
    crop/extract a subregion form an nd image.
    """
    dim = len(volume.shape)
    min_idx = start_idx.astype('int64')
    max_idx = min_idx + np.array(output_size).astype('int64')

    assert (dim >= 2 and dim <= 5)
    # assert (max_idx[0] - min_idx[0] <= volume.shape[0])
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
