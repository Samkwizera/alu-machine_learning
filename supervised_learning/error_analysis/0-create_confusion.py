#!/usr/bin/env python3
"""Create a confusion matrix."""

import numpy as np


def create_confusion_matrix(labels, logits):
    """Create a confusion matrix from one-hot labels and predictions."""
    classes = labels.shape[1]
    actual = np.argmax(labels, axis=1)
    predicted = np.argmax(logits, axis=1)
    confusion = np.zeros((classes, classes))

    np.add.at(confusion, (actual, predicted), 1)
    return confusion
