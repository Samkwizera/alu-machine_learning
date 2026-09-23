#!/usr/bin/env python3
"""Trains a transformer for Portuguese to English machine translation."""

import tensorflow.compat.v2 as tf
Dataset = __import__('3-dataset').Dataset
create_masks = __import__('4-create_masks').create_masks
Transformer = __import__('5-transformer').Transformer


class Schedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Learning rate schedule with a linear warmup and inverse sqrt decay."""

    def __init__(self, dm, warmup_steps=4000):
        """
        dm is the dimensionality of the model
        warmup_steps is the number of warmup steps
        """
        super(Schedule, self).__init__()
        self.dm = tf.cast(dm, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        """Returns the learning rate for the given step."""
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.dm) * tf.math.minimum(arg1, arg2)


def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):
    """
    Creates and trains a transformer model for machine translation

    N is the number of blocks in the encoder and decoder
    dm is the dimensionality of the model
    h is the number of heads
    hidden is the number of hidden units in the fully connected layers
    max_len is the maximum number of tokens per sequence
    batch_size is the batch size for training
    epochs is the number of epochs to train for

    Returns: the trained model
    """
    data = Dataset(batch_size, max_len)
    input_vocab = data.tokenizer_pt.vocab_size + 2
    target_vocab = data.tokenizer_en.vocab_size + 2

    transformer = Transformer(N, dm, h, hidden, input_vocab, target_vocab,
                              max_len, max_len)

    learning_rate = Schedule(dm)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9,
                                         beta_2=0.98, epsilon=1e-9)

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')

    def loss_function(real, pred):
        """Calculates the loss, ignoring the padded tokens."""
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss = loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_accuracy')

    @tf.function
    def train_step(inputs, target):
        """Runs a single training step on one batch."""
        target_inputs = target[:, :-1]
        target_real = target[:, 1:]

        encoder_mask, combined_mask, decoder_mask = create_masks(
            inputs, target_inputs)

        with tf.GradientTape() as tape:
            predictions = transformer(inputs, target_inputs, True,
                                      encoder_mask, combined_mask,
                                      decoder_mask)
            loss = loss_function(target_real, predictions)

        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(
            zip(gradients, transformer.trainable_variables))

        train_loss(loss)
        train_accuracy(target_real, predictions)

    for epoch in range(epochs):
        train_loss.reset_states()
        train_accuracy.reset_states()

        for batch, (inputs, target) in enumerate(data.data_train):
            train_step(inputs, target)

            if batch % 50 == 0:
                print('Epoch {}, batch {}: loss {} accuracy {}'.format(
                    epoch + 1, batch, train_loss.result(),
                    train_accuracy.result()))

        print('Epoch {}: loss {} accuracy {}'.format(
            epoch + 1, train_loss.result(), train_accuracy.result()))

    return transformer
