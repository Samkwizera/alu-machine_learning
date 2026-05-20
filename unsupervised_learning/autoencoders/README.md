# Autoencoders

This directory contains TensorFlow Keras exercises for building autoencoders.
The models learn compressed representations of input data and reconstruct the
original input from those representations.

## Files

- `0-vanilla.py`: creates a fully connected vanilla autoencoder.
- `1-sparse.py`: creates a sparse autoencoder with L1 regularization.
- `2-convolutional.py`: creates a convolutional autoencoder.
- `3-variational.py`: creates a variational autoencoder.

## Requirements

- Ubuntu 16.04 LTS
- Python 3.5
- TensorFlow 1.x
- NumPy
- Matplotlib for the example visualization scripts

Each Python file defines an `autoencoder` function that returns the encoder,
decoder, and full autoencoder models.
