#!/usr/bin/env python3
"""Defines a deep neural network."""

import numpy as np


class DeepNeuralNetwork:
    """Deep neural network performing binary classification."""

    def __init__(self, nx, layers):
        """Initialize a deep neural network."""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        prev = nx
        for i, nodes in enumerate(layers, 1):
            if not isinstance(nodes, int) or nodes <= 0:
                raise TypeError("layers must be a list of positive integers")
            self.__weights["W{}".format(i)] = (
                np.random.randn(nodes, prev) * np.sqrt(2 / prev)
            )
            self.__weights["b{}".format(i)] = np.zeros((nodes, 1))
            prev = nodes

    @property
    def L(self):
        """Number of layers."""
        return self.__L

    @property
    def cache(self):
        """Intermediary values."""
        return self.__cache

    @property
    def weights(self):
        """Network weights and biases."""
        return self.__weights
