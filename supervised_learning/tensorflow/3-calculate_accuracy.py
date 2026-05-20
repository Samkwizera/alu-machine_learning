#!/usr/bin/env python3
"""Calculate prediction accuracy for a TensorFlow neural network."""

import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """Calculate the accuracy of a prediction.

    Args:
        y: Placeholder for one-hot labels.
        y_pred: Tensor containing the network predictions.

    Returns:
        A tensor containing the decimal accuracy.
    """
    correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    return tf.reduce_mean(tf.cast(correct, tf.float32))
