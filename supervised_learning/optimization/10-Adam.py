#!/usr/bin/env python3
"""Create a TensorFlow Adam optimization operation."""

import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """Return an Adam optimizer training operation."""
    optimizer = tf.train.AdamOptimizer(
        alpha, beta1=beta1, beta2=beta2, epsilon=epsilon)
    return optimizer.minimize(loss)
