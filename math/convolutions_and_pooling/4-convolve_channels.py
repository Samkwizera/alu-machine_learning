#!/usr/bin/env python3
"""Convolution on images with channels."""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """Perform a convolution on images with channels."""
    m, h, w, c = images.shape
    kh, kw, _ = kernel.shape
    sh, sw = stride

    if padding == 'same':
        ph = max((kh - 1) // 2, kh // 2)
        pw = max((kw - 1) // 2, kw // 2)
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    out_h = (h + 2 * ph - kh) // sh + 1
    out_w = (w + 2 * pw - kw) // sw + 1

    padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                    mode='constant')

    output = np.zeros((m, out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            patch = padded[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :]
            output[:, i, j] = np.sum(patch * kernel, axis=(1, 2, 3))

    return output
