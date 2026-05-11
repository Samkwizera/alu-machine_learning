#!/usr/bin/env python3
"""Defines a single neuron for binary classification."""

import numpy as np


class Neuron:
    """Single neuron performing binary classification."""

    def __init__(self, nx):
        """Initialize a neuron.

        Args:
            nx: Number of input features.

        Raises:
            TypeError: If nx is not an integer.
            ValueError: If nx is less than 1.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.W = np.random.randn(1, nx)
        self.b = 0
        self.A = 0
