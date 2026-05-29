#!/usr/bin/env python3
"""Conduct forward propagation with dropout regularization."""

import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """Return layer activations and dropout masks for forward propagation."""
    cache = {'A0': X}

    for layer in range(1, L + 1):
        W = weights['W{}'.format(layer)]
        b = weights['b{}'.format(layer)]
        A_prev = cache['A{}'.format(layer - 1)]
        Z = np.matmul(W, A_prev) + b

        if layer == L:
            t = np.exp(Z)
            cache['A{}'.format(layer)] = t / np.sum(
                t, axis=0, keepdims=True)
        else:
            A = np.tanh(Z)
            D = np.random.binomial(1, keep_prob, size=A.shape)
            A = (A * D) / keep_prob
            cache['D{}'.format(layer)] = D
            cache['A{}'.format(layer)] = A

    return cache
