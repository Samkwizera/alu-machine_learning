#!/usr/bin/env python3
"""Initialize centroids for K-means clustering."""

import numpy as np


def initialize(X, k):
    """Initialize ``k`` centroids uniformly within the bounds of ``X``."""
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or
            X.shape[0] == 0 or X.shape[1] == 0 or
            not isinstance(k, int) or isinstance(k, bool) or k <= 0):
        return None

    return np.random.uniform(np.min(X, axis=0), np.max(X, axis=0),
                             size=(k, X.shape[1]))
