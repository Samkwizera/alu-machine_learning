#!/usr/bin/env python3
"""Create a TensorFlow inverse time learning rate decay operation."""

import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """Return a stepwise inverse-time decay operation."""
    return tf.train.inverse_time_decay(
        alpha, global_step, decay_step, decay_rate, staircase=True)
