#!/usr/bin/env python3
"""Choose actions using an epsilon-greedy policy."""

import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """Return an exploratory or greedy action for ``state``."""
    if np.random.uniform(0, 1) > epsilon:
        return np.argmax(Q[state])
    return np.random.randint(Q.shape[1])
