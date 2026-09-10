#!/usr/bin/env python3
"""Forward propagation for a deep recurrent neural network."""

import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """Perform forward propagation through a deep RNN."""
    time_steps, batch_size, _ = X.shape
    layers, _, hidden_size = h_0.shape
    output_size = rnn_cells[-1].by.shape[1]

    H = np.zeros((time_steps + 1, layers, batch_size, hidden_size))
    Y = np.zeros((time_steps, batch_size, output_size))
    H[0] = h_0

    for step in range(time_steps):
        layer_input = X[step]
        for layer, cell in enumerate(rnn_cells):
            hidden, output = cell.forward(H[step, layer], layer_input)
            H[step + 1, layer] = hidden
            layer_input = hidden
        Y[step] = output

    return H, Y
