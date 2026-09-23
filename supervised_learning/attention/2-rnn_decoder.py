#!/usr/bin/env python3
"""RNN decoder for machine translation."""

import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """Decodes one target word at a time using attention."""

    def __init__(self, vocab, embedding, units, batch):
        """
        vocab is the size of the output vocabulary
        embedding is the dimensionality of the embedding vector
        units is the number of hidden units in the RNN cell
        batch is the batch size
        """
        super(RNNDecoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(units,
                                       recurrent_initializer='glorot_uniform',
                                       return_sequences=True,
                                       return_state=True)
        self.F = tf.keras.layers.Dense(vocab)

    def call(self, x, s_prev, hidden_states):
        """
        x is a tensor of shape (batch, 1) of the previous target word
        s_prev is a tensor of shape (batch, units) of the previous state
        hidden_states is a tensor of shape (batch, input_seq_len, units)

        Returns: y, s
        """
        units = s_prev.get_shape().as_list()[1]
        attention = SelfAttention(units)
        context, _ = attention(s_prev, hidden_states)

        x = self.embedding(x)
        context = tf.expand_dims(context, 1)
        x = tf.concat([context, x], axis=-1)

        outputs, s = self.gru(x)
        outputs = tf.reshape(outputs, (-1, outputs.shape[2]))
        y = self.F(outputs)
        return y, s
