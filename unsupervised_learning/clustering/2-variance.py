#!/usr/bin/env python3
"""Calculate total intra-cluster variance."""

import numpy as np


def variance(X, C):
    """Return the sum of squared distances to the nearest centroid."""
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or
            not isinstance(C, np.ndarray) or C.ndim != 2 or
            X.shape[0] == 0 or C.shape[0] == 0 or
            X.shape[1] != C.shape[1]):
        return None

    distances = X[:, np.newaxis, :] - C[np.newaxis, :, :]
    squared = np.sum(distances ** 2, axis=2)
    return np.sum(np.min(squared, axis=1))
