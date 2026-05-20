#!/usr/bin/env python3
"""Calculate loss for a TensorFlow neural network."""

import tensorflow as tf


def calculate_loss(y, y_pred):
    """Calculate softmax cross-entropy loss.

    Args:
        y: Placeholder for one-hot labels.
        y_pred: Tensor containing the network predictions.

    Returns:
        A tensor containing the loss.
    """
    return tf.losses.softmax_cross_entropy(y, y_pred)
