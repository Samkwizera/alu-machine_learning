#!/usr/bin/env python3
"""Randomly shear an image."""

import tensorflow as tf


def shear_image(image, intensity):
    """Randomly shear an image with the given intensity."""
    return tf.keras.preprocessing.image.random_shear(
        image, intensity, row_axis=0, col_axis=1, channel_axis=2
    )
