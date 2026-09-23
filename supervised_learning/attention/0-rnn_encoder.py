#!/usr/bin/env python3
"""RNN encoder for machine translation."""

import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """Encodes a sequence of word indices into hidden states."""

    def __init__(self, vocab, embedding, units, batch):
        """
        vocab is the size of the input vocabulary
        embedding is the dimensionality of the embedding vector
        units is the number of hidden units in the RNN cell
        batch is the batch size
        """
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(units,
                                       recurrent_initializer='glorot_uniform',
                                       return_sequences=True,
                                       return_state=True)

    def initialize_hidden_state(self):
        """Returns a tensor of zeros of shape (batch, units)."""
        return tf.zeros((self.batch, self.units))

    def call(self, x, initial):
        """
        x is a tensor of shape (batch, input_seq_len) of word indices
        initial is a tensor of shape (batch, units) of the initial state

        Returns: outputs, hidden
        """
        x = self.embedding(x)
        outputs, hidden = self.gru(x, initial_state=initial)
        return outputs, hidden
