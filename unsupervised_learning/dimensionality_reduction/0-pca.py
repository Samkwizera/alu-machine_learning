#!/usr/bin/env python3
"""Principal component analysis based on retained variance."""

import numpy as np


def pca(X, var=0.95):
    """Return the PCA weight matrix that retains ``var`` of the variance."""
    _, singular_values, right_vectors = np.linalg.svd(
        X, full_matrices=False)
    cumulative_variance = np.cumsum(singular_values)
    cumulative_variance /= np.sum(singular_values)
    ndim = np.searchsorted(cumulative_variance, var) + 1
    return right_vectors[:ndim].T
