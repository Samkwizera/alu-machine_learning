#!/usr/bin/env python3
"""Build, train, and save a batch-normalized neural network."""

import tensorflow as tf

create_batch_norm_layer = __import__(
    '14-batch_norm').create_batch_norm_layer
shuffle_data = __import__('2-shuffle_data').shuffle_data


def model(Data_train, Data_valid, layers, activations, alpha=0.001,
          beta1=0.9, beta2=0.999, epsilon=1e-8, decay_rate=1,
          batch_size=32, epochs=5, save_path='/tmp/model.ckpt'):
    """Build, train, and save a neural network model."""
    X_train, Y_train = Data_train
    X_valid, Y_valid = Data_valid

    x = tf.placeholder(tf.float32, shape=(None, X_train.shape[1]), name='x')
    y = tf.placeholder(tf.float32, shape=(None, Y_train.shape[1]), name='y')

    y_pred = x
    for layer, activation in zip(layers, activations):
        y_pred = create_batch_norm_layer(y_pred, layer, activation)

    loss = tf.losses.softmax_cross_entropy(y, y_pred)
    correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.inverse_time_decay(
        alpha, global_step, 1, decay_rate, staircase=True)
    train_op = tf.train.AdamOptimizer(
        learning_rate, beta1=beta1, beta2=beta2,
        epsilon=epsilon).minimize(loss)
    increment_step = tf.assign_add(global_step, 1)

    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    tf.add_to_collection('y_pred', y_pred)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('train_op', train_op)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(epochs + 1):
            train_cost, train_accuracy = sess.run(
                [loss, accuracy], feed_dict={x: X_train, y: Y_train})
            valid_cost, valid_accuracy = sess.run(
                [loss, accuracy], feed_dict={x: X_valid, y: Y_valid})

            print('After {} epochs:'.format(epoch))
            print('\tTraining Cost: {}'.format(train_cost))
            print('\tTraining Accuracy: {}'.format(train_accuracy))
            print('\tValidation Cost: {}'.format(valid_cost))
            print('\tValidation Accuracy: {}'.format(valid_accuracy))

            if epoch == epochs:
                break

            X_shuffle, Y_shuffle = shuffle_data(X_train, Y_train)
            step = 0

            for start in range(0, X_train.shape[0], batch_size):
                end = start + batch_size
                X_batch = X_shuffle[start:end]
                Y_batch = Y_shuffle[start:end]
                step += 1

                sess.run(train_op, feed_dict={x: X_batch, y: Y_batch})

                if step % 100 == 0:
                    step_cost, step_accuracy = sess.run(
                        [loss, accuracy],
                        feed_dict={x: X_batch, y: Y_batch})
                    print('\tStep {}:'.format(step))
                    print('\t\tCost: {}'.format(step_cost))
                    print('\t\tAccuracy: {}'.format(step_accuracy))

            sess.run(increment_step)

        return saver.save(sess, save_path)
