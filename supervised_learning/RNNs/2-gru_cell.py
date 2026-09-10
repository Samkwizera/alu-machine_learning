#!/usr/bin/env python3
"""A gated recurrent unit cell."""

import numpy as np


class GRUCell:
    """Represent one gated recurrent unit cell."""

    def __init__(self, i, h, o):
        """Initialize the cell's weights and biases."""
        self.Wz = np.random.randn(i + h, h)
        self.Wr = np.random.randn(i + h, h)
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    @staticmethod
    def sigmoid(value):
        """Return the sigmoid activation of value."""
        return 1 / (1 + np.exp(-value))

    def forward(self, h_prev, x_t):
        """Perform forward propagation for one time step."""
        cell_input = np.concatenate((h_prev, x_t), axis=1)
        z = self.sigmoid(np.matmul(cell_input, self.Wz) + self.bz)
        r = self.sigmoid(np.matmul(cell_input, self.Wr) + self.br)

        candidate_input = np.concatenate((r * h_prev, x_t), axis=1)
        h_inter = np.tanh(np.matmul(candidate_input, self.Wh) + self.bh)
        h_next = (1 - z) * h_prev + z * h_inter

        output = np.matmul(h_next, self.Wy) + self.by
        output -= np.max(output, axis=1, keepdims=True)
        exp_output = np.exp(output)
        y = exp_output / np.sum(exp_output, axis=1, keepdims=True)
        return h_next, y
