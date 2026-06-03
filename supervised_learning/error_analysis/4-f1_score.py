#!/usr/bin/env python3
"""Calculate F1 score for each class in a confusion matrix."""

sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """Return the F1 score of each class."""
    recall = sensitivity(confusion)
    prec = precision(confusion)

    return 2 * ((prec * recall) / (prec + recall))
