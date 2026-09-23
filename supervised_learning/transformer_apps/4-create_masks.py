#!/usr/bin/env python3
"""Creates all masks for training/validation."""

import tensorflow.compat.v2 as tf


def create_masks(inputs, target):
    """
    inputs is a tf.Tensor of shape (batch_size, seq_len_in)
    target is a tf.Tensor of shape (batch_size, seq_len_out)

    Returns: encoder_mask, combined_mask, decoder_mask
    """
    batch_size, seq_len_out = target.shape

    encoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    encoder_mask = encoder_mask[:, tf.newaxis, tf.newaxis, :]

    decoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    decoder_mask = decoder_mask[:, tf.newaxis, tf.newaxis, :]

    look_ahead_mask = 1 - tf.linalg.band_part(
        tf.ones((seq_len_out, seq_len_out)), -1, 0)
    target_mask = tf.cast(tf.math.equal(target, 0), tf.float32)
    target_mask = target_mask[:, tf.newaxis, tf.newaxis, :]
    combined_mask = tf.maximum(target_mask, look_ahead_mask)

    return encoder_mask, combined_mask, decoder_mask
