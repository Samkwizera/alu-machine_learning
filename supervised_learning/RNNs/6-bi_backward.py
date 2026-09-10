#!/usr/bin/env python3
"""A bidirectional recurrent neural network cell."""

import numpy as np


class BidirectionalCell:
    """Represent one cell of a bidirectional recurrent neural network."""

    def __init__(self, i, h, o):
        """Initialize the cell's weights and biases."""
        self.Whf = np.random.randn(i + h, h)
        self.Whb = np.random.randn(i + h, h)
        self.Wy = np.random.randn(2 * h, o)
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Calculate the forward-direction hidden state."""
        cell_input = np.concatenate((h_prev, x_t), axis=1)
        return np.tanh(np.matmul(cell_input, self.Whf) + self.bhf)

    def backward(self, h_next, x_t):
        """Calculate the backward-direction hidden state."""
        cell_input = np.concatenate((h_next, x_t), axis=1)
        return np.tanh(np.matmul(cell_input, self.Whb) + self.bhb)
