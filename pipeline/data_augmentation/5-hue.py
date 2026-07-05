#!/usr/bin/env python3
"""Change image hue."""

import tensorflow as tf


def change_hue(image, delta):
    """Change the hue of an image by delta."""
    return tf.image.adjust_hue(image, delta)
