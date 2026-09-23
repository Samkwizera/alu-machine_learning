#!/usr/bin/env python3
"""A full transformer network for machine translation."""

import numpy as np
import tensorflow.compat.v2 as tf


def positional_encoding(max_seq_len, dm):
    """
    Calculates the positional encoding for a transformer

    max_seq_len is the maximum sequence length
    dm is the model depth

    Returns: a numpy.ndarray of shape (max_seq_len, dm)
    """
    PE = np.zeros((max_seq_len, dm))
    position = np.arange(max_seq_len)[:, np.newaxis]
    index = np.arange(0, dm, 2)
    div_term = np.power(10000, index / dm)

    PE[:, 0::2] = np.sin(position / div_term)
    PE[:, 1::2] = np.cos(position / div_term)
    return PE


def sdp_attention(Q, K, V, mask=None):
    """
    Calculates the scaled dot product attention

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
        Q, K, and V are the inputs used to generate the query, key, and
        value matrices, and mask is the mask to apply

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


class EncoderBlock(tf.keras.layers.Layer):
    """Creates an encoder block for a transformer."""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        dm is the dimensionality of the model
        h is the number of heads
        hidden is the number of hidden units in the fully connected layer
        drop_rate is the dropout rate
        """
        super(EncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """
        x is a tensor of shape (batch, input_seq_len, dm)

        Returns: a tensor of shape (batch, input_seq_len, dm)
        """
        attention, _ = self.mha(x, x, x, mask)
        attention = self.dropout1(attention, training=training)
        out1 = self.layernorm1(x + attention)

        hidden = self.dense_hidden(out1)
        output = self.dense_output(hidden)
        output = self.dropout2(output, training=training)
        return self.layernorm2(out1 + output)


class DecoderBlock(tf.keras.layers.Layer):
    """Creates a decoder block for a transformer."""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        dm is the dimensionality of the model
        h is the number of heads
        hidden is the number of hidden units in the fully connected layer
        drop_rate is the dropout rate
        """
        super(DecoderBlock, self).__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask,
             padding_mask):
        """
        x is a tensor of shape (batch, target_seq_len, dm)
        encoder_output is a tensor of shape (batch, input_seq_len, dm)

        Returns: a tensor of shape (batch, target_seq_len, dm)
        """
        attention1, _ = self.mha1(x, x, x, look_ahead_mask)
        attention1 = self.dropout1(attention1, training=training)
        out1 = self.layernorm1(x + attention1)

        attention2, _ = self.mha2(out1, encoder_output, encoder_output,
                                  padding_mask)
        attention2 = self.dropout2(attention2, training=training)
        out2 = self.layernorm2(out1 + attention2)

        hidden = self.dense_hidden(out2)
        output = self.dense_output(hidden)
        output = self.dropout3(output, training=training)
        return self.layernorm3(out2 + output)


class Encoder(tf.keras.layers.Layer):
    """Creates the encoder for a transformer."""

    def __init__(self, N, dm, h, hidden, input_vocab, max_seq_len,
                 drop_rate=0.1):
        """
        N is the number of blocks in the encoder
        dm is the dimensionality of the model
        h is the number of heads
        hidden is the number of hidden units in the fully connected layer
        input_vocab is the size of the input vocabulary
        max_seq_len is the maximum sequence length possible
        drop_rate is the dropout rate
        """
        super(Encoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [EncoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """
        x is a tensor of shape (batch, input_seq_len)

        Returns: a tensor of shape (batch, input_seq_len, dm)
        """
        seq_len = x.shape[1]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += tf.cast(self.positional_encoding[:seq_len], tf.float32)
        x = self.dropout(x, training=training)

        for i in range(self.N):
            x = self.blocks[i](x, training, mask)

        return x


class Decoder(tf.keras.layers.Layer):
    """Creates the decoder for a transformer."""

    def __init__(self, N, dm, h, hidden, target_vocab, max_seq_len,
                 drop_rate=0.1):
        """
        N is the number of blocks in the decoder
        dm is the dimensionality of the model
        h is the number of heads
        hidden is the number of hidden units in the fully connected layer
        target_vocab is the size of the target vocabulary
        max_seq_len is the maximum sequence length possible
        drop_rate is the dropout rate
        """
        super(Decoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [DecoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask,
             padding_mask):
        """
        x is a tensor of shape (batch, target_seq_len)
        encoder_output is a tensor of shape (batch, input_seq_len, dm)

        Returns: a tensor of shape (batch, target_seq_len, dm)
        """
        seq_len = x.shape[1]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += tf.cast(self.positional_encoding[:seq_len], tf.float32)
        x = self.dropout(x, training=training)

        for i in range(self.N):
            x = self.blocks[i](x, encoder_output, training, look_ahead_mask,
                               padding_mask)

        return x


class Transformer(tf.keras.Model):
    """Creates a transformer network."""

    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab,
                 max_seq_input, max_seq_target, drop_rate=0.1):
        """
        N is the number of blocks in the encoder and decoder
        dm is the dimensionality of the model
        h is the number of heads
        hidden is the number of hidden units in the fully connected layers
        input_vocab is the size of the input vocabulary
        target_vocab is the size of the target vocabulary
        max_seq_input is the maximum input sequence length possible
        max_seq_target is the maximum target sequence length possible
        drop_rate is the dropout rate
        """
        super(Transformer, self).__init__()
        self.encoder = Encoder(N, dm, h, hidden, input_vocab, max_seq_input,
                               drop_rate)
        self.decoder = Decoder(N, dm, h, hidden, target_vocab, max_seq_target,
                               drop_rate)
        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(self, inputs, target, training, encoder_mask, look_ahead_mask,
             decoder_mask):
        """
        inputs is a tensor of shape (batch, input_seq_len)
        target is a tensor of shape (batch, target_seq_len)

        Returns: a tensor of shape (batch, target_seq_len, target_vocab)
        """
        encoder_output = self.encoder(inputs, training, encoder_mask)
        decoder_output = self.decoder(target, encoder_output, training,
                                      look_ahead_mask, decoder_mask)
        return self.linear(decoder_output)
