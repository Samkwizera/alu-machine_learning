#!/usr/bin/env python3
"""Scaled dot product attention."""

import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """
    Q is a tensor with last two dimensions (..., seq_len_q, dk)
    K is a tensor with last two dimensions (..., seq_len_v, dk)
    V is a tensor with last two dimensions (..., seq_len_v, dv)
    mask is broadcastable into (..., seq_len_q, seq_len_v) or None

    Returns: output, weights
    """
    matmul_qk = tf.matmul(Q, K, transpose_b=True)
    dk = tf.cast(tf.shape(K)[-1], matmul_qk.dtype)
    scaled = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled += (mask * -1e9)

    weights = tf.nn.softmax(scaled, axis=-1)
    output = tf.matmul(weights, V)
    return output, weights
