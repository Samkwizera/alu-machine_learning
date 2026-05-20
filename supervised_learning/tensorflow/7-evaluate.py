#!/usr/bin/env python3
"""Evaluate a saved TensorFlow neural network."""

import tensorflow as tf


def evaluate(X, Y, save_path):
    """Evaluate the output of a saved neural network.

    Args:
        X: Input data to evaluate.
        Y: One-hot labels for X.
        save_path: Location to load the model from.

    Returns:
        The prediction, accuracy, and loss, respectively.
    """
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(save_path + '.meta')
        saver.restore(sess, save_path)
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        y_pred = tf.get_collection('y_pred')[0]
        loss = tf.get_collection('loss')[0]
        accuracy = tf.get_collection('accuracy')[0]
        return sess.run(
            [y_pred, accuracy, loss], feed_dict={x: X, y: Y})
