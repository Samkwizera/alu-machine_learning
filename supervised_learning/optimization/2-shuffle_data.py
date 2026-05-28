#!/usr/bin/env python3
"""Shuffle two data matrices in the same way."""

import numpy as np


def shuffle_data(X, Y):
    """Return X and Y shuffled with the same permutation."""
    permutation = np.random.permutation(X.shape[0])
    return X[permutation], Y[permutation]
