#!/usr/bin/env python3
"""Update variables using gradient descent with momentum."""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """Return the updated variable and first moment."""
    v = (beta1 * v) + ((1 - beta1) * grad)
    var = var - (alpha * v)

    return var, v
