#!/usr/bin/env python3
"""One-hot encode labels."""

import numpy as np


def one_hot_encode(Y, classes):
    """Convert a label vector into a one-hot matrix."""
    if not isinstance(Y, np.ndarray) or not isinstance(classes, int):
        return None
    if classes <= 0 or len(Y.shape) != 1:
        return None
    try:
        one_hot = np.zeros((classes, Y.shape[0]))
        one_hot[Y, np.arange(Y.shape[0])] = 1
        return one_hot
    except Exception:
        return None
