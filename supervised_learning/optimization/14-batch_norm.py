#!/usr/bin/env python3
"""Create a TensorFlow batch normalization layer."""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """Return the activated output of a batch normalization layer."""
    initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG")
    dense = tf.layers.Dense(n, kernel_initializer=initializer)
    Z = dense(prev)

    gamma = tf.Variable(tf.ones([1, n]))
    beta = tf.Variable(tf.zeros([1, n]))
    mean, variance = tf.nn.moments(Z, axes=[0])
    Z_norm = tf.nn.batch_normalization(
        Z, mean, variance, beta, gamma, 1e-8)

    if activation is None:
        return Z_norm
    return activation(Z_norm)
