#!/usr/bin/env python3
"""Multivariate Gaussian probability density function."""

import numpy as np


def pdf(X, m, S):
    """Evaluate a multivariate normal distribution at every row of ``X``."""
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or
            not isinstance(m, np.ndarray) or m.ndim != 1 or
            not isinstance(S, np.ndarray) or S.ndim != 2 or
            X.shape[0] == 0 or X.shape[1] != m.shape[0] or
            S.shape != (m.shape[0], m.shape[0])):
        return None

    determinant = np.linalg.det(S)
    if not np.isfinite(determinant) or determinant <= 0:
        return None

    try:
        inverse = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        return None

    difference = X - m
    exponent = np.einsum('ni,ij,nj->n', difference, inverse, difference)
    denominator = np.sqrt(((2 * np.pi) ** X.shape[1]) * determinant)
    P = np.exp(-0.5 * exponent) / denominator
    return np.maximum(P, 1e-300)
