#!/usr/bin/env python3
"""Gaussian process update module."""

import numpy as np


class GaussianProcess:
    """Represents a noiseless 1D Gaussian process."""

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """Initialize a Gaussian process."""
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """Calculate the RBF covariance kernel matrix."""
        X1_sq = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
        X2_sq = np.sum(X2 ** 2, axis=1)
        sqdist = X1_sq + X2_sq - 2 * np.matmul(X1, X2.T)
        return (self.sigma_f ** 2) * np.exp(-0.5 / (self.l ** 2) * sqdist)

    def predict(self, X_s):
        """Predict the mean and variance of points in a Gaussian process."""
        K_s = self.kernel(X_s, self.X)
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(self.K)

        mu = np.matmul(np.matmul(K_s, K_inv), self.Y).reshape(-1)
        sigma = np.diag(K_ss - np.matmul(np.matmul(K_s, K_inv), K_s.T))

        return mu, sigma

    def update(self, X_new, Y_new):
        """Update a Gaussian process with a new sample."""
        self.X = np.concatenate((self.X, X_new.reshape(1, 1)), axis=0)
        self.Y = np.concatenate((self.Y, Y_new.reshape(1, 1)), axis=0)
        self.K = self.kernel(self.X, self.X)
