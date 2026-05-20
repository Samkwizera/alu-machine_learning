#!/usr/bin/env python3
"""Create a layer for a TensorFlow neural network."""

import tensorflow as tf


def create_layer(prev, n, activation):
    """Create a fully connected layer.

    Args:
        prev: Tensor output of the previous layer.
        n: Number of nodes in the layer to create.
        activation: Activation function for the layer.

    Returns:
        The tensor output of the layer.
    """
    initializer = tf.contrib.layers.variance_scaling_initializer(
        mode='FAN_AVG')
    layer = tf.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer,
        name='layer'
    )
    return layer(prev)
