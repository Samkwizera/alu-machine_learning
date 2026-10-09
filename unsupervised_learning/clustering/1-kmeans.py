#!/usr/bin/env python3
"""K-means clustering."""

import numpy as np


def kmeans(X, k, iterations=1000):
    """Cluster ``X`` into ``k`` groups using Lloyd's algorithm."""
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or
            X.shape[0] == 0 or X.shape[1] == 0 or
            not isinstance(k, int) or isinstance(k, bool) or k <= 0 or
            not isinstance(iterations, int) or isinstance(iterations, bool) or
            iterations <= 0):
        return None, None

    low = np.min(X, axis=0)
    high = np.max(X, axis=0)
    C = np.random.uniform(low, high, size=(k, X.shape[1]))

    for _ in range(iterations):
        distances = np.linalg.norm(X[:, np.newaxis, :] - C, axis=2)
        clss = np.argmin(distances, axis=1)
        previous = C.copy()

        counts = np.bincount(clss, minlength=k)
        sums = np.zeros_like(C, dtype=float)
        np.add.at(sums, clss, X)
        nonempty = counts != 0
        C[nonempty] = sums[nonempty] / counts[nonempty, np.newaxis]

        empty = ~nonempty
        if np.any(empty):
            replacements = np.random.uniform(low, high,
                                             size=(np.sum(empty), X.shape[1]))
            C[empty] = replacements

        if np.array_equal(C, previous):
            break

    distances = np.linalg.norm(X[:, np.newaxis, :] - C, axis=2)
    clss = np.argmin(distances, axis=1)
    return C, clss
