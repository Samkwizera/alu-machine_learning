#!/usr/bin/env python3
"""Convolution on images using multiple kernels."""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """Perform a convolution on images using multiple kernels."""
    m, h, w, c = images.shape
    kh, kw, _, nc = kernels.shape
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

    output = np.zeros((m, out_h, out_w, nc))

    for i in range(out_h):
        for j in range(out_w):
            for k in range(nc):
                patch = padded[
                    :, i * sh:i * sh + kh, j * sw:j * sw + kw, :]
                output[:, i, j, k] = np.sum(
                    patch * kernels[:, :, :, k], axis=(1, 2, 3)
                )

    return output
