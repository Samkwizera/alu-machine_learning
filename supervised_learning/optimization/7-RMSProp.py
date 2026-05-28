#!/usr/bin/env python3
"""Update variables using the RMSProp optimization algorithm."""

import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """Return the updated variable and second moment."""
    s = (beta2 * s) + ((1 - beta2) * (grad ** 2))
    var = var - (alpha * grad / (np.sqrt(s) + epsilon))

    return var, s
