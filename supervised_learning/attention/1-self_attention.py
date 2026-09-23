#!/usr/bin/env python3
"""Self attention for machine translation."""

import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """Calculates the attention for machine translation."""

    def __init__(self, units):
        """units is the number of hidden units in the alignment model"""
        super(SelfAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """
        s_prev is a tensor of shape (batch, units) of the previous
            decoder hidden state
        hidden_states is a tensor of shape (batch, input_seq_len, units)
            of the encoder outputs

        Returns: context, weights
        """
        s_expanded = tf.expand_dims(s_prev, 1)
        score = self.V(tf.nn.tanh(self.W(s_expanded) + self.U(hidden_states)))
        weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(weights * hidden_states, axis=1)
        return context, weights
