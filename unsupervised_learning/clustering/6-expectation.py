#!/usr/bin/env python3
"""Expectation step of the EM algorithm for a GMM."""

import numpy as np

pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """Calculate posterior cluster probabilities and log likelihood."""
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or
            not isinstance(pi, np.ndarray) or pi.ndim != 1 or
            not isinstance(m, np.ndarray) or m.ndim != 2 or
            not isinstance(S, np.ndarray) or S.ndim != 3 or
            X.shape[0] == 0 or X.shape[1] == 0 or
            m.shape != (pi.shape[0], X.shape[1]) or
            S.shape != (pi.shape[0], X.shape[1], X.shape[1]) or
            np.any(pi < 0) or not np.isclose(np.sum(pi), 1)):
        return None, None

    g = np.empty((pi.shape[0], X.shape[0]))
    for cluster in range(pi.shape[0]):
        probability = pdf(X, m[cluster], S[cluster])
        if probability is None:
            return None, None
        g[cluster] = pi[cluster] * probability

    marginal = np.sum(g, axis=0)
    if np.any(marginal <= 0) or not np.all(np.isfinite(marginal)):
        return None, None
    likelihood = np.sum(np.log(marginal))
    g /= marginal
    return g, likelihood
