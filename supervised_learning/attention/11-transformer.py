#!/usr/bin/env python3
"""Transformer network."""

import tensorflow as tf
Encoder = __import__('9-transformer_encoder').Encoder
Decoder = __import__('10-transformer_decoder').Decoder


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
        training is a boolean for whether the model is training
        encoder_mask is the padding mask for the encoder
        look_ahead_mask is the look ahead mask for the decoder
        decoder_mask is the padding mask for the decoder

        Returns: a tensor of shape (batch, target_seq_len, target_vocab)
        """
        encoder_output = self.encoder(inputs, training, encoder_mask)
        decoder_output = self.decoder(target, encoder_output, training,
                                      look_ahead_mask, decoder_mask)
        return self.linear(decoder_output)
