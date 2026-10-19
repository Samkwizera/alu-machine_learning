#!/usr/bin/env python3
"""Determine whether a Markov chain is absorbing."""

import numpy as np


def absorbing(P):
    """Return whether every state can reach an absorbing state."""
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return False
    if P.shape[0] == 0 or P.shape[0] != P.shape[1]:
        return False
    if not np.all(np.isfinite(P)) or np.any(P < 0):
        return False
    if not np.allclose(P.sum(axis=1), 1):
        return False

    n = P.shape[0]
    absorbing_states = [i for i in range(n) if P[i, i] == 1]
    if not absorbing_states:
        return False

    reachable = set(absorbing_states)
    stack = list(absorbing_states)
    while stack:
        destination = stack.pop()
        for source in np.flatnonzero(P[:, destination] > 0):
            source = int(source)
            if source not in reachable:
                reachable.add(source)
                stack.append(source)
    return len(reachable) == n
