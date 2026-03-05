#!/usr/bin/env python3
"""Pooling on images."""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """Perform pooling on images."""
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    out_h = (h - kh) // sh + 1
    out_w = (w - kw) // sw + 1

    output = np.zeros((m, out_h, out_w, c))

    for i in range(out_h):
        for j in range(out_w):
            patch = images[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :]
            if mode == 'max':
                output[:, i, j, :] = np.max(patch, axis=(1, 2))
            else:
                output[:, i, j, :] = np.mean(patch, axis=(1, 2))

    return output
