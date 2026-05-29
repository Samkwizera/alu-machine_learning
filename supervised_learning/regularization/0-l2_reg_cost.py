#!/usr/bin/env python3
"""Calculate a neural network cost with L2 regularization."""

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """Return the cost of a neural network with L2 regularization."""
    l2_norm = 0

    for layer in range(1, L + 1):
        l2_norm += np.sum(np.square(weights['W{}'.format(layer)]))

    return cost + ((lambtha * l2_norm) / (2 * m))
