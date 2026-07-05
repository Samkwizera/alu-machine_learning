#!/usr/bin/env python3
"""Randomly crop an image."""

import tensorflow as tf


def crop_image(image, size):
    """Perform a random crop of an image."""
    return tf.image.random_crop(image, size)
