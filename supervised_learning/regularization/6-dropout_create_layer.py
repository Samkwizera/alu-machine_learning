#!/usr/bin/env python3
"""Create a TensorFlow layer with dropout regularization."""

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """Return a TensorFlow layer that uses dropout."""
    initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG")
    layer = tf.layers.Dense(
        n, activation=activation, kernel_initializer=initializer)
    dropout = tf.layers.Dropout(1 - keep_prob)

    return dropout(layer(prev))
