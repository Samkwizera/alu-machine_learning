#!/usr/bin/env python3
"""Find the steady state of a regular Markov chain."""

import numpy as np


def regular(P):
    """Return the steady-state distribution of a regular chain."""
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return None
    if P.shape[0] == 0 or P.shape[0] != P.shape[1]:
        return None
    if not np.all(np.isfinite(P)) or np.any(P < 0):
        return None
    if not np.allclose(P.sum(axis=1), 1):
        return None

    n = P.shape[0]
    power = np.eye(n)
    # A nonnegative n x n primitive matrix has a positive power no later
    # than Wielandt's bound.
    bound = 1 if n == 1 else (n - 1) ** 2 + 1
    is_regular = False
    for _ in range(bound):
        power = np.matmul(power, P)
        if np.all(power > 0):
            is_regular = True
            break
    if not is_regular:
        return None

    system = np.vstack((P.T - np.eye(n), np.ones((1, n))))
    target = np.append(np.zeros(n), 1)
    steady = np.linalg.lstsq(system, target, rcond=None)[0]
    steady[np.isclose(steady, 0)] = 0
    if np.any(steady < 0):
        return None
    steady /= steady.sum()
    return steady.reshape(1, n)
