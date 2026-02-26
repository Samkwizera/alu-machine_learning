#!/usr/bin/env python3
"""Calculates the posterior probability for various hypothetical probabilities
given the data."""
import numpy as np
intersection = __import__('1-intersection').intersection
marginal = __import__('2-marginal').marginal


def posterior(x, n, P, Pr):
    """Calculates the posterior probability for each probability in P.

    x: number of patients that develop severe side effects
    n: total number of patients observed
    P: 1D numpy.ndarray of hypothetical probabilities
    Pr: 1D numpy.ndarray of prior beliefs of P
    Returns: 1D numpy.ndarray of posterior probabilities for each p in P
    """
    return intersection(x, n, P, Pr) / marginal(x, n, P, Pr)
