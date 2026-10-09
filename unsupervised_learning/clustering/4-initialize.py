#!/usr/bin/env python3
"""Initialize the parameters of a Gaussian mixture model."""

import numpy as np

kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """Return equal priors, K-means means, and identity covariances."""
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or
            X.shape[0] == 0 or X.shape[1] == 0 or
            not isinstance(k, int) or isinstance(k, bool) or k <= 0):
        return None, None, None

    pi = np.full(k, 1 / k)
    m, _ = kmeans(X, k)
    S = np.tile(np.identity(X.shape[1]), (k, 1, 1))
    return pi, m, S
