#!/usr/bin/env python3
"""A long short-term memory cell."""

import numpy as np


class LSTMCell:
    """Represent one long short-term memory cell."""

    def __init__(self, i, h, o):
        """Initialize the cell's weights and biases."""
        self.Wf = np.random.randn(i + h, h)
        self.Wu = np.random.randn(i + h, h)
        self.Wc = np.random.randn(i + h, h)
        self.Wo = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    @staticmethod
    def sigmoid(value):
        """Return the sigmoid activation of value."""
        return 1 / (1 + np.exp(-value))

    def forward(self, h_prev, c_prev, x_t):
        """Perform forward propagation for one time step."""
        cell_input = np.concatenate((h_prev, x_t), axis=1)
        forget = self.sigmoid(np.matmul(cell_input, self.Wf) + self.bf)
        update = self.sigmoid(np.matmul(cell_input, self.Wu) + self.bu)
        candidate = np.tanh(np.matmul(cell_input, self.Wc) + self.bc)
        output_gate = self.sigmoid(
            np.matmul(cell_input, self.Wo) + self.bo
        )

        c_next = forget * c_prev + update * candidate
        h_next = output_gate * np.tanh(c_next)

        output = np.matmul(h_next, self.Wy) + self.by
        output -= np.max(output, axis=1, keepdims=True)
        exp_output = np.exp(output)
        y = exp_output / np.sum(exp_output, axis=1, keepdims=True)
        return h_next, c_next, y
