import numpy as np
import torch
import torch.nn as nn


class Panchors3D(nn.Module):
    def __init__(self, drift=0.5, down_levels=None, strides=None):
        super(Panchors3D, self).__init__()
        self.drift = drift
        if down_levels is None:
            self.down_levels = [4]
        else:
            self.down_levels = down_levels
        self.strides = self.down_levels

    def forward(self, image):

        image_shape = image.shape[2:]
        image_shape = np.array(image_shape)
        image_shapes = [(image_shape + x - 1) // x for x in self.down_levels]
        bs = image.shape[0]

        # compute anchors over all pyramid levels
        all_panchors = []

        for idx, p in enumerate(self.down_levels):
            anchors = generate_panchors_3d()
            shifted_panchors = shift(image_shapes[idx], self.strides[idx], anchors, drift=self.drift)
            shifted_panchors = torch.unsqueeze(shifted_panchors, dim=0).to(image.device)
            all_panchors.append(shifted_panchors)

        all_panchors = torch.cat(all_panchors, dim=1)
        all_panchors = all_panchors.repeat(bs, 1, 1)
        return all_panchors


def generate_panchors_3d():
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales w.r.t. a reference window.
    """

    # (z_ctr, y_ctr, x_ctr)
    panchors = torch.zeros((1, 3))

    return panchors


def shift(shape, stride, anchors, drift=0.5):
    shift_z = (torch.arange(0, shape[0]) + drift) * stride
    shift_y = (torch.arange(0, shape[1]) + drift) * stride
    shift_x = (torch.arange(0, shape[2]) + drift) * stride

    shift_z, shift_y, shift_x = torch.meshgrid(shift_z, shift_y, shift_x)

    shifts = torch.stack((shift_z, shift_y, shift_x), dim=-1).float()

    # add A panchors (1, A, 3) to
    # cell K shifts (K, 1, 3) to get
    # shift anchors (K, A, 3)
    # reshape to (K*A, 3) shifted anchors
    A = anchors.shape[0]
    K = shifts.shape[0] * shifts.shape[1] * shifts.shape[2]
    all_anchors = (anchors.reshape((1, A, 3)) + shifts.reshape((1, K, 3)).permute((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 3))

    return all_anchors
