#!/usr/bin/env python3
"""Determine useful K-means cluster counts by variance."""

import numpy as np

kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """Run K-means over a range and return variance improvements."""
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or
            X.shape[0] == 0 or X.shape[1] == 0 or
            not isinstance(kmin, int) or isinstance(kmin, bool) or kmin <= 0 or
            (kmax is not None and
             (not isinstance(kmax, int) or isinstance(kmax, bool))) or
            not isinstance(iterations, int) or isinstance(iterations, bool) or
            iterations <= 0):
        return None, None

    if kmax is None:
        kmax = X.shape[0]
    if kmax <= kmin:
        return None, None

    results = []
    variances = []
    for k in range(kmin, kmax + 1):
        result = kmeans(X, k, iterations)
        results.append(result)
        variances.append(variance(X, result[0]))

    baseline = variances[0]
    d_vars = [baseline - current for current in variances]
    return results, d_vars
