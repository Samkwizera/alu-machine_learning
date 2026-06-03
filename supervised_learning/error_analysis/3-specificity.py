#!/usr/bin/env python3
"""Calculate specificity for each class in a confusion matrix."""

import numpy as np


def specificity(confusion):
    """Return the specificity of each class."""
    true_positives = np.diag(confusion)
    actual_positives = np.sum(confusion, axis=1)
    predicted_positives = np.sum(confusion, axis=0)
    total = np.sum(confusion)
    true_negatives = total - actual_positives - predicted_positives
    true_negatives += true_positives
    actual_negatives = total - actual_positives

    return true_negatives / actual_negatives
