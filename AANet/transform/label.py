from __future__ import print_function, division

from .abstract_transform import AbstractTransform
import numpy as np
from scipy import ndimage
import skimage.measure
import skimage.morphology
from scipy.ndimage.morphology import binary_closing, binary_dilation, generate_binary_structure
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


class LabelToAnnot(AbstractTransform):
    def __init__(self, min_area=2):
        """
        """
        self.min_area = min_area

    def __call__(self, sample):
        label = sample['label']
        label = skimage.measure.label(label[0])
        label = skimage.morphology.remove_small_objects(label, self.min_area)

        regions = skimage.measure.regionprops(label)
        obj_bbox = np.array([region.bbox for region in regions]).reshape(-1, 6)
        obj_area = np.array([region.area for region in regions])
        obj_id = np.array([region.label for region in regions])

        obj_cls = np.zeros([obj_bbox.shape[0], 1])
        annot = np.concatenate([obj_bbox, obj_cls], axis=-1)

        sample['annot'] = annot
        sample['label'] = np.expand_dims((label > 0), axis=0)

        return sample


class LabelClose(AbstractTransform):
    def __init__(self):
        """
        """
        self.struct = generate_binary_structure(3, 2)

    def __call__(self, sample):
        label = sample['label']
        dlabel = binary_closing(label[0], self.struct)
        label[0] = dlabel
        sample['label'] = label

        return sample


class RemoveSmall(AbstractTransform):
    def __init__(self, min_area=2):
        """
        """
        self.min_area = min_area

    def __call__(self, sample):
        label = sample['label']
        label = skimage.measure.label(label[0])
        label = skimage.morphology.remove_small_objects(label, self.min_area)
        sample['label'] = np.expand_dims((label > 0), axis=0)

        return sample


class MaskToLabel(AbstractTransform):
    def __init__(self):
        pass

    def __call__(self, sample):
        sample['label'] = np.concatenate([sample['label'], sample['mask']])

        return sample


class MaskToImage(AbstractTransform):
    def __init__(self):
        pass

    def __call__(self, sample):
        sample['image'] = np.concatenate([sample['image'], sample['mask']])

        return sample