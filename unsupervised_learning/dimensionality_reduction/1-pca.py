#!/usr/bin/env python3
"""Principal component analysis with a fixed output dimensionality."""

import numpy as np


def pca(X, ndim):
    """Transform ``X`` into ``ndim`` principal components."""
    centered = X - np.mean(X, axis=0)
    _, _, right_vectors = np.linalg.svd(centered, full_matrices=False)
    weights = right_vectors[:ndim].T
    return np.matmul(centered, weights)
