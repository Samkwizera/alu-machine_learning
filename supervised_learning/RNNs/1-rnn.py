#!/usr/bin/env python3
"""Forward propagation for a simple recurrent neural network."""

import numpy as np


def rnn(rnn_cell, X, h_0):
    """Perform forward propagation through a simple RNN."""
    time_steps, batch_size, _ = X.shape
    hidden_size = h_0.shape[1]
    output_size = rnn_cell.by.shape[1]

    H = np.zeros((time_steps + 1, batch_size, hidden_size))
    Y = np.zeros((time_steps, batch_size, output_size))
    H[0] = h_0

    for step in range(time_steps):
        H[step + 1], Y[step] = rnn_cell.forward(H[step], X[step])

    return H, Y
