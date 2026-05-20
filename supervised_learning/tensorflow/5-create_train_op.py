#!/usr/bin/env python3
"""Create the training operation for a TensorFlow neural network."""

import tensorflow as tf


def create_train_op(loss, alpha):
    """Create the training operation using gradient descent.

    Args:
        loss: Loss tensor for the network prediction.
        alpha: Learning rate.

    Returns:
        An operation that trains the network.
    """
    optimizer = tf.train.GradientDescentOptimizer(alpha)
    return optimizer.minimize(loss)
