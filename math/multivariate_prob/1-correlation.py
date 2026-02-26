#!/usr/bin/env python3
"""Function to calculate a correlation matrix from a covariance matrix."""
import numpy as np


def correlation(C):
    """Calculate a correlation matrix.

    Args:
        C: numpy.ndarray of shape (d, d) containing a covariance matrix

    Returns:
        numpy.ndarray of shape (d, d) containing the correlation matrix
    """
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    std = np.sqrt(np.diag(C))
    corr = C / np.outer(std, std)

    return corr
