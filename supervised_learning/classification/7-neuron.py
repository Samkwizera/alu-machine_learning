#!/usr/bin/env python3
"""Defines a single neuron for binary classification."""

import matplotlib.pyplot as plt
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

    def forward_prop(self, X):
        """Calculate forward propagation."""
        self.__A = 1 / (1 + np.exp(-(np.matmul(self.__W, X) + self.__b)))
        return self.__A

    def cost(self, Y, A):
        """Calculate logistic regression cost."""
        m = Y.shape[1]
        return -np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)) / m

    def evaluate(self, X, Y):
        """Evaluate the neuron's predictions."""
        A = self.forward_prop(X)
        return np.where(A >= 0.5, 1, 0), self.cost(Y, A)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Perform one pass of gradient descent."""
        m = Y.shape[1]
        dz = A - Y
        dw = np.matmul(dz, X.T) / m
        db = np.sum(dz) / m
        self.__W = self.__W - alpha * dw
        self.__b = self.__b - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """Train the neuron."""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        steps = []
        costs = []
        for i in range(iterations + 1):
            A = self.forward_prop(X)
            if (verbose or graph) and (i % step == 0 or i == iterations):
                cost = self.cost(Y, A)
                steps.append(i)
                costs.append(cost)
                if verbose:
                    print("Cost after {} iterations: {}".format(i, cost))
            if i < iterations:
                self.gradient_descent(X, Y, A, alpha)

        if graph:
            plt.plot(steps, costs, "b")
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()
        return self.evaluate(X, Y)
