#!/usr/bin/env python3
"""Create a variational autoencoder."""

import tensorflow.keras as keras
from tensorflow.keras import backend as K


def sampling(args):
    """Sample a latent vector from a mean and log variance."""
    mean, log_var = args
    epsilon = K.random_normal(shape=K.shape(mean))
    return mean + K.exp(log_var / 2) * epsilon


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Create a variational autoencoder.

    Args:
        input_dims: Dimensions of the model input.
        hidden_layers: Nodes in each encoder hidden layer.
        latent_dims: Dimensions of the latent space representation.

    Returns:
        The encoder, decoder, and variational autoencoder models.
    """
    encoder_input = keras.Input(shape=(input_dims,))
    encoded = encoder_input
    for nodes in hidden_layers:
        encoded = keras.layers.Dense(nodes, activation='relu')(encoded)
    mean = keras.layers.Dense(latent_dims)(encoded)
    log_var = keras.layers.Dense(latent_dims)(encoded)
    latent = keras.layers.Lambda(sampling)([mean, log_var])
    encoder = keras.Model(encoder_input, [latent, mean, log_var])

    decoder_input = keras.Input(shape=(latent_dims,))
    decoded = decoder_input
    for nodes in hidden_layers[::-1]:
        decoded = keras.layers.Dense(nodes, activation='relu')(decoded)
    decoder_output = keras.layers.Dense(
        input_dims, activation='sigmoid')(decoded)
    decoder = keras.Model(decoder_input, decoder_output)

    auto_output = decoder(encoder(encoder_input)[0])
    auto = keras.Model(encoder_input, auto_output)
    reconstruction_loss = keras.losses.binary_crossentropy(
        encoder_input, auto_output)
    reconstruction_loss *= input_dims
    kl_loss = -0.5 * K.sum(
        1 + log_var - K.square(mean) - K.exp(log_var), axis=1)
    auto.add_loss(K.mean(reconstruction_loss + kl_loss))
    auto.compile(optimizer='adam')

    return encoder, decoder, auto
