#!/usr/bin/env python3
"""Multi head attention."""

import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """Performs multi head attention."""

    def __init__(self, dm, h):
        """
        dm is the dimensionality of the model
        h is the number of heads, dm is divisible by h
        """
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // h
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def split_heads(self, x, batch):
        """Splits the last dimension into (h, depth) and transposes."""
        x = tf.reshape(x, (batch, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """
        Q is a tensor of shape (batch, seq_len_q, dk)
        K is a tensor of shape (batch, seq_len_v, dk)
        V is a tensor of shape (batch, seq_len_v, dv)
        mask is always None

        Returns: output, weights
        """
        batch = tf.shape(Q)[0]

        q = self.split_heads(self.Wq(Q), batch)
        k = self.split_heads(self.Wk(K), batch)
        v = self.split_heads(self.Wv(V), batch)

        attention, weights = sdp_attention(q, k, v, mask)

        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        attention = tf.reshape(attention, (batch, -1, self.dm))

        output = self.linear(attention)
        return output, weights
