#!/usr/bin/env python3
"""Select a Gaussian mixture model using Bayesian information criterion."""

import numpy as np

expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """Fit a range of GMMs and return the model with the lowest BIC."""
    failure = (None, None, None, None)
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or
            X.shape[0] == 0 or X.shape[1] == 0 or
            not isinstance(kmin, int) or isinstance(kmin, bool) or kmin <= 0 or
            (kmax is not None and
             (not isinstance(kmax, int) or isinstance(kmax, bool))) or
            not isinstance(iterations, int) or isinstance(iterations, bool) or
            iterations <= 0 or not isinstance(tol, float) or tol < 0 or
            not isinstance(verbose, bool)):
        return failure

    if kmax is None:
        kmax = X.shape[0]
    if kmax < kmin:
        return failure

    cluster_counts = range(kmin, kmax + 1)
    likelihoods = np.empty(kmax - kmin + 1)
    criteria = np.empty(kmax - kmin + 1)
    results = []
    dimensions = X.shape[1]

    for index, clusters in enumerate(cluster_counts):
        result = expectation_maximization(
            X, clusters, iterations, tol, verbose)
        if result[0] is None:
            return failure
        pi, m, S, _, likelihood = result
        parameters = (clusters * dimensions +
                      clusters * dimensions * (dimensions + 1) / 2 +
                      clusters - 1)
        likelihoods[index] = likelihood
        criteria[index] = parameters * np.log(X.shape[0]) - 2 * likelihood
        results.append((pi, m, S))

    best_index = np.argmin(criteria)
    best_k = kmin + best_index
    return best_k, results[best_index], likelihoods, criteria
