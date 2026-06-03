#!/usr/bin/env python3
"""Calculate precision for each class in a confusion matrix."""

import numpy as np


def precision(confusion):
    """Return the precision of each class."""
    true_positives = np.diag(confusion)
    predicted_positives = np.sum(confusion, axis=0)

    return true_positives / predicted_positives
