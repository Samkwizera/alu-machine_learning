#!/usr/bin/env python3
"""Maximization step of the EM algorithm for a GMM."""

import numpy as np


def maximization(X, g):
    """Update GMM priors, means, and covariance matrices."""
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or
            not isinstance(g, np.ndarray) or g.ndim != 2 or
            X.shape[0] == 0 or X.shape[1] == 0 or
            g.shape[1] != X.shape[0] or g.shape[0] == 0 or
            np.any(g < 0) or not np.allclose(np.sum(g, axis=0), 1)):
        return None, None, None

    totals = np.sum(g, axis=1)
    if np.any(totals == 0):
        return None, None, None

    pi = totals / X.shape[0]
    m = np.matmul(g, X) / totals[:, np.newaxis]
    S = np.empty((g.shape[0], X.shape[1], X.shape[1]))
    for cluster in range(g.shape[0]):
        difference = X - m[cluster]
        S[cluster] = np.matmul((difference * g[cluster, :, np.newaxis]).T,
                               difference) / totals[cluster]
    return pi, m, S
