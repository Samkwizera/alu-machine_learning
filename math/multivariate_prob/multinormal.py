#!/usr/bin/env python3
"""Class representing a Multivariate Normal distribution."""
import numpy as np


class MultiNormal:
    """Represents a Multivariate Normal distribution."""

    def __init__(self, data):
        """Initialize MultiNormal.

        Args:
            data: numpy.ndarray of shape (d, n) containing the data set
        """
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = np.mean(data, axis=1, keepdims=True)
        deviation = data - self.mean
        self.cov = (deviation @ deviation.T) / (n - 1)

    def pdf(self, x):
        """Calculate the PDF at a data point.

        Args:
            x: numpy.ndarray of shape (d, 1) containing the data point

        Returns:
            float: the value of the PDF at x
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")
        d = self.mean.shape[0]
        if x.shape != (d, 1):
            raise ValueError("x must have the shape ({}, 1)".format(d))

        det = np.linalg.det(self.cov)
        inv = np.linalg.inv(self.cov)
        diff = x - self.mean
        exponent = -0.5 * (diff.T @ inv @ diff)[0, 0]
        coefficient = 1.0 / (np.sqrt(((2 * np.pi) ** d) * det))

        return coefficient * np.exp(exponent)
