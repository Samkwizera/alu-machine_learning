#!/usr/bin/env python3
"""Strided convolution on grayscale images."""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """Perform a convolution on grayscale images."""
    m, h, w = images.shape
    kh, kw = kernel.shape
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

    padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), mode='constant')

    output = np.zeros((m, out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            output[:, i, j] = np.sum(
                padded[:, i * sh:i * sh + kh, j * sw:j * sw + kw] * kernel,
                axis=(1, 2)
            )

    return output
