#!/usr/bin/env python3
"""Calculates the marginal probability of obtaining the data."""
import numpy as np
intersection = __import__('1-intersection').intersection


def marginal(x, n, P, Pr):
    """Calculates the marginal probability of obtaining x and n.

    x: number of patients that develop severe side effects
    n: total number of patients observed
    P: 1D numpy.ndarray of hypothetical probabilities
    Pr: 1D numpy.ndarray of prior beliefs of P
    Returns: the marginal probability of obtaining x and n
    """
    return np.sum(intersection(x, n, P, Pr))
