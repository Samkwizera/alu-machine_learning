#!/usr/bin/env python3
"""Create a TensorFlow momentum optimization operation."""

import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """Return a momentum optimizer training operation."""
    optimizer = tf.train.MomentumOptimizer(alpha, beta1)
    return optimizer.minimize(loss)
