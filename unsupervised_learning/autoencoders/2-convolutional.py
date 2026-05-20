#!/usr/bin/env python3
"""Create a convolutional autoencoder."""

import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """Create a convolutional autoencoder.

    Args:
        input_dims: Dimensions of the model input.
        filters: Number of filters for each encoder convolutional layer.
        latent_dims: Dimensions of the latent space representation.

    Returns:
        The encoder, decoder, and autoencoder models.
    """
    encoder_input = keras.Input(shape=input_dims)
    encoded = encoder_input
    for n_filters in filters:
        encoded = keras.layers.Conv2D(
            n_filters, (3, 3), padding='same', activation='relu')(encoded)
        encoded = keras.layers.MaxPooling2D((2, 2), padding='same')(encoded)
    encoder = keras.Model(encoder_input, encoded)

    decoder_input = keras.Input(shape=latent_dims)
    decoded = decoder_input
    decoder_filters = filters[::-1]
    for n_filters in decoder_filters[:-1]:
        decoded = keras.layers.Conv2D(
            n_filters, (3, 3), padding='same', activation='relu')(decoded)
        decoded = keras.layers.UpSampling2D((2, 2))(decoded)
    decoded = keras.layers.Conv2D(
        decoder_filters[-1], (3, 3), padding='valid',
        activation='relu')(decoded)
    decoded = keras.layers.UpSampling2D((2, 2))(decoded)
    decoder_output = keras.layers.Conv2D(
        input_dims[-1], (3, 3), padding='same',
        activation='sigmoid')(decoded)
    decoder = keras.Model(decoder_input, decoder_output)

    auto_output = decoder(encoder(encoder_input))
    auto = keras.Model(encoder_input, auto_output)
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
