#!/usr/bin/env python3
"""The forward algorithm for a hidden Markov model."""

import numpy as np


def _valid_model(Observation, Emission, Transition, Initial):
    """Validate the arrays that describe a hidden Markov model."""
    if not isinstance(Observation, np.ndarray) or Observation.ndim != 1:
        return False
    if Observation.size == 0 or not np.issubdtype(Observation.dtype,
                                                  np.integer):
        return False
    if not isinstance(Emission, np.ndarray) or Emission.ndim != 2:
        return False
    if not isinstance(Transition, np.ndarray) or Transition.ndim != 2:
        return False
    if not isinstance(Initial, np.ndarray) or Initial.ndim != 2:
        return False
    n, m = Emission.shape
    if n == 0 or m == 0 or Transition.shape != (n, n):
        return False
    if Initial.shape != (n, 1):
        return False
    arrays = (Emission, Transition, Initial)
    if any(not np.all(np.isfinite(array)) for array in arrays):
        return False
    if any(np.any(array < 0) for array in arrays):
        return False
    if (not np.allclose(Emission.sum(axis=1), 1) or
            not np.allclose(Transition.sum(axis=1), 1) or
            not np.isclose(Initial.sum(), 1)):
        return False
    return bool(np.all((Observation >= 0) & (Observation < m)))


def forward(Observation, Emission, Transition, Initial):
    """Return the observation likelihood and forward path probabilities."""
    if not _valid_model(Observation, Emission, Transition, Initial):
        return None, None

    n = Emission.shape[0]
    t = Observation.size
    F = np.zeros((n, t))
    F[:, 0] = Initial[:, 0] * Emission[:, Observation[0]]
    for step in range(1, t):
        F[:, step] = (F[:, step - 1] @ Transition) * \
            Emission[:, Observation[step]]
    return np.sum(F[:, -1]), F
