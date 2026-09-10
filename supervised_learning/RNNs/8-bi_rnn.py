#!/usr/bin/env python3
"""Forward propagation for a bidirectional recurrent neural network."""

import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """Perform forward propagation through a bidirectional RNN."""
    time_steps, batch_size, _ = X.shape
    hidden_size = h_0.shape[1]
    forward_states = np.zeros((time_steps, batch_size, hidden_size))
    backward_states = np.zeros((time_steps, batch_size, hidden_size))

    h_prev = h_0
    for step in range(time_steps):
        h_prev = bi_cell.forward(h_prev, X[step])
        forward_states[step] = h_prev

    h_next = h_t
    for step in range(time_steps - 1, -1, -1):
        h_next = bi_cell.backward(h_next, X[step])
        backward_states[step] = h_next

    H = np.concatenate((forward_states, backward_states), axis=2)
    Y = bi_cell.output(H)
    return H, Y
