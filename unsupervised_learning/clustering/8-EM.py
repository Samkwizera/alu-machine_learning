#!/usr/bin/env python3
"""Expectation-maximization algorithm for a Gaussian mixture model."""

import numpy as np

initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5,
                             verbose=False):
    """Fit a ``k``-component Gaussian mixture model to ``X``."""
    failure = (None, None, None, None, None)
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or
            X.shape[0] == 0 or X.shape[1] == 0 or
            not isinstance(k, int) or isinstance(k, bool) or k <= 0 or
            not isinstance(iterations, int) or isinstance(iterations, bool) or
            iterations <= 0 or
            not isinstance(tol, float) or tol < 0 or
            not isinstance(verbose, bool)):
        return failure

    pi, m, S = initialize(X, k)
    if pi is None:
        return failure
    g, likelihood = expectation(X, pi, m, S)
    if g is None:
        return failure
    if verbose:
        print('Log Likelihood after 0 iterations: {:.5f}'.format(likelihood))

    for iteration in range(1, iterations + 1):
        pi, m, S = maximization(X, g)
        if pi is None:
            return failure
        new_g, new_likelihood = expectation(X, pi, m, S)
        if new_g is None:
            return failure
        converged = abs(new_likelihood - likelihood) <= tol
        g, likelihood = new_g, new_likelihood
        if verbose and (iteration % 10 == 0 or converged or
                        iteration == iterations):
            print('Log Likelihood after {} iterations: {:.5f}'.format(
                iteration, likelihood))
        if converged:
            break

    return pi, m, S, g, likelihood
