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

    def forward_prop(self, X):
        """Calculate forward propagation."""
        self.__cache["A0"] = X
        for i in range(1, self.__L + 1):
            z = (np.matmul(self.__weights["W{}".format(i)],
                           self.__cache["A{}".format(i - 1)]) +
                 self.__weights["b{}".format(i)])
            self.__cache["A{}".format(i)] = 1 / (1 + np.exp(-z))
        return self.__cache["A{}".format(self.__L)], self.__cache

    def cost(self, Y, A):
        """Calculate logistic regression cost."""
        m = Y.shape[1]
        return -np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)) / m

    def evaluate(self, X, Y):
        """Evaluate the neural network's predictions."""
        A, _ = self.forward_prop(X)
        return np.where(A >= 0.5, 1, 0), self.cost(Y, A)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Perform one pass of gradient descent."""
        m = Y.shape[1]
        weights = self.__weights.copy()
        dz = cache["A{}".format(self.__L)] - Y
        for i in range(self.__L, 0, -1):
            w_key = "W{}".format(i)
            b_key = "b{}".format(i)
            a_prev = cache["A{}".format(i - 1)]
            dw = np.matmul(dz, a_prev.T) / m
            db = np.sum(dz, axis=1, keepdims=True) / m
            if i > 1:
                dz = np.matmul(weights[w_key].T, dz)
                dz = dz * a_prev * (1 - a_prev)
            self.__weights[w_key] = weights[w_key] - alpha * dw
            self.__weights[b_key] = weights[b_key] - alpha * db
