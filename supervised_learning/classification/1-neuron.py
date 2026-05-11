#!/usr/bin/env python3
"""Defines a single neuron for binary classification."""

import numpy as np


class Neuron:
    """Single neuron performing binary classification."""

    def __init__(self, nx):
        """Initialize a neuron."""
        if not isinstance(nx, int):
            raise TypeError("nx must be a integer")
        if nx < 1:
            raise ValueError("nx must be positive")

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Weights vector."""
        return self.__W

    @property
    def b(self):
        """Bias."""
        return self.__b

    @property
    def A(self):
        """Activated output."""
        return self.__A
