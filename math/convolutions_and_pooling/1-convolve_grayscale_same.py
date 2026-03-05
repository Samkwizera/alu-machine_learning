#!/usr/bin/env python3
import numpy as np


def convolve_grayscale_same(images, kernel):
    """Performs a same convolution on grayscale images.

    Args:
        images: numpy.ndarray shape (m, h, w) - grayscale images
        kernel: numpy.ndarray shape (kh, kw) - convolution kernel

    Returns:
        numpy.ndarray containing the convolved images (same size as input)
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    ph = max((kh - 1) // 2, kh // 2)
    pw = max((kw - 1) // 2, kw // 2)

    padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), mode='constant')

    output = np.zeros((m, h, w))

    for i in range(h):
        for j in range(w):
            output[:, i, j] = np.sum(
                padded[:, i:i + kh, j:j + kw] * kernel, axis=(1, 2)
            )

    return output
