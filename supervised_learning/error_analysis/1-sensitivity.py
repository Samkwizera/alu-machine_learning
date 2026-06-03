#!/usr/bin/env python3
"""Calculate sensitivity for each class in a confusion matrix."""

import numpy as np


def sensitivity(confusion):
    """Return the sensitivity of each class."""
    true_positives = np.diag(confusion)
    actual_positives = np.sum(confusion, axis=1)

    return true_positives / actual_positives
