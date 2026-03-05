#!/usr/bin/env python3
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """Performs a convolution on grayscale images with custom padding.

    Args:
        images: numpy.ndarray shape (m, h, w) - grayscale images
        kernel: numpy.ndarray shape (kh, kw) - convolution kernel
        padding: tuple (ph, pw) - padding for height and width

    Returns:
        numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    out_h = h + 2 * ph - kh + 1
    out_w = w + 2 * pw - kw + 1

    padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), mode='constant')

    output = np.zeros((m, out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            output[:, i, j] = np.sum(
                padded[:, i:i + kh, j:j + kw] * kernel, axis=(1, 2)
            )

    return output
