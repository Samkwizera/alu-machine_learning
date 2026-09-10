#!/usr/bin/env python3
"""A simple recurrent neural network cell."""

import numpy as np


class RNNCell:
    """Represent one cell of a simple recurrent neural network."""

    def __init__(self, i, h, o):
        """Initialize the cell's weights and biases."""
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Perform forward propagation for one time step."""
        cell_input = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(cell_input, self.Wh) + self.bh)
        output = np.matmul(h_next, self.Wy) + self.by
        output -= np.max(output, axis=1, keepdims=True)
        exp_output = np.exp(output)
        y = exp_output / np.sum(exp_output, axis=1, keepdims=True)
        return h_next, y
