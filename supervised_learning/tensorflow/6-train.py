#!/usr/bin/env python3
"""Build, train, and save a TensorFlow neural network classifier."""

import tensorflow as tf

calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations,
          alpha, iterations, save_path='/tmp/model.ckpt'):
    """Build, train, and save a neural network classifier.

    Args:
        X_train: Training input data.
        Y_train: Training one-hot labels.
        X_valid: Validation input data.
        Y_valid: Validation one-hot labels.
        layer_sizes: Number of nodes in each layer.
        activations: Activation functions for each layer.
        alpha: Learning rate.
        iterations: Number of training iterations.
        save_path: Location where the model should be saved.

    Returns:
        The path where the model was saved.
    """
    nx = X_train.shape[1]
    classes = Y_train.shape[1]
    x, y = create_placeholders(nx, classes)
    y_pred = forward_prop(x, layer_sizes, activations)
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    train_op = create_train_op(loss, alpha)

    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    tf.add_to_collection('y_pred', y_pred)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('train_op', train_op)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(iterations + 1):
            if i == 0 or i % 100 == 0 or i == iterations:
                train_cost, train_accuracy = sess.run(
                    [loss, accuracy], feed_dict={x: X_train, y: Y_train})
                valid_cost, valid_accuracy = sess.run(
                    [loss, accuracy], feed_dict={x: X_valid, y: Y_valid})
                print('After {} iterations:'.format(i))
                print('\tTraining Cost: {}'.format(train_cost))
                print('\tTraining Accuracy: {}'.format(train_accuracy))
                print('\tValidation Cost: {}'.format(valid_cost))
                print('\tValidation Accuracy: {}'.format(valid_accuracy))
            if i < iterations:
                sess.run(train_op, feed_dict={x: X_train, y: Y_train})
        return saver.save(sess, save_path)
