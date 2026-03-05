#!/usr/bin/env python3
"""
Function that performs a convolution on grayscale images
"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Function that performs a convolution on grayscale images
    """
    m, h, w = images.shape[0], images.shape[1], images.shape[2]
    kh, kw = kernel.shape[0], kernel.shape[1]
    sh, sw = stride[0], stride[1]
    if padding == "same":
        ph = int(((w - 1) * sw + kw - w) / 2) + 1
        pw = int(((h - 1) * sh + kh - h) / 2) + 1
    elif padding == "valid":
        ph = 0
        pw = 0
    else:
        pw = padding[1]
        ph = padding[0]
    nw = int(((w - kw + (2 * pw)) / sw) + 1)
    nh = int(((h - kh + (2 * ph)) / sh) + 1)
    convolved = np.zeros((m, nh, nw))
    npad = ((0, 0), (ph, ph), (pw, pw))
    imagesp = np.pad(
        images,
        pad_width=npad,
        mode="constant",
        constant_values=0)
    for i in range(nh):
        x = i * stride[0]
        for j in range(nw):
            y = j * sw
            image = imagesp[:, x: x + kh, y: y + kw]
            convolved[:, i, j] = np.sum(
                np.multiply(image, kernel), axis=(1, 2))
    return convolved
