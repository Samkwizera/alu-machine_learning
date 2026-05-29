#!/usr/bin/env python3
"""Calculate a TensorFlow cost with L2 regularization."""

import tensorflow as tf


def l2_reg_cost(cost):
    """Return the TensorFlow cost accounting for L2 regularization."""
    return cost + tf.losses.get_regularization_losses()
