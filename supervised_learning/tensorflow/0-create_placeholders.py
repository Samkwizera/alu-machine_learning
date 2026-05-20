#!/usr/bin/env python3
"""Create placeholders for a TensorFlow neural network."""

import tensorflow as tf


def create_placeholders(nx, classes):
    """Create placeholders for input data and one-hot labels.

    Args:
        nx: Number of feature columns in the data.
        classes: Number of classes in the classifier.

    Returns:
        The placeholders x and y.
    """
    x = tf.placeholder(tf.float32, shape=(None, nx), name='x')
    y = tf.placeholder(tf.float32, shape=(None, classes), name='y')
    return x, y
