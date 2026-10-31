#!/usr/bin/env python3
"""Initialize Q-tables."""

import numpy as np


def q_init(env):
    """Return a zero-filled Q-table for an environment's state/action space."""
    return np.zeros((env.observation_space.n, env.action_space.n))
