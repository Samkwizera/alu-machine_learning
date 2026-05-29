#!/usr/bin/env python3
"""Update neural network parameters with L2 regularization."""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """Update weights and biases using gradient descent with L2."""
    m = Y.shape[1]
    dZ = cache['A{}'.format(L)] - Y

    for layer in range(L, 0, -1):
        W_key = 'W{}'.format(layer)
        b_key = 'b{}'.format(layer)
        A_prev = cache['A{}'.format(layer - 1)]
        W = weights[W_key]

        dW = (np.matmul(dZ, A_prev.T) / m) + ((lambtha / m) * W)
        db = np.sum(dZ, axis=1, keepdims=True) / m

        if layer > 1:
            dA_prev = np.matmul(W.T, dZ)
            dZ = dA_prev * (1 - np.square(A_prev))

        weights[W_key] = W - (alpha * dW)
        weights[b_key] = weights[b_key] - (alpha * db)
