#!/usr/bin/env python3
"""Create a TensorFlow RMSProp optimization operation."""

import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """Return an RMSProp optimizer training operation."""
    optimizer = tf.train.RMSPropOptimizer(alpha, decay=beta2, epsilon=epsilon)
    return optimizer.minimize(loss)
