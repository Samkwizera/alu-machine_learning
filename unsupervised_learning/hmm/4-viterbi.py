#!/usr/bin/env python3
"""The Viterbi algorithm for a hidden Markov model."""

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


def viterbi(Observation, Emission, Transition, Initial):
    """Return the most likely hidden-state path and its probability."""
    if not _valid_model(Observation, Emission, Transition, Initial):
        return None, None

    n = Emission.shape[0]
    t = Observation.size
    scores = np.zeros((n, t))
    previous = np.zeros((n, t), dtype=int)
    scores[:, 0] = Initial[:, 0] * Emission[:, Observation[0]]

    for step in range(1, t):
        candidates = scores[:, step - 1, None] * Transition
        previous[:, step] = np.argmax(candidates, axis=0)
        scores[:, step] = np.max(candidates, axis=0) * \
            Emission[:, Observation[step]]

    last = int(np.argmax(scores[:, -1]))
    probability = scores[last, -1]
    path = [last]
    for step in range(t - 1, 0, -1):
        path.append(int(previous[path[-1], step]))
    path.reverse()
    return path, probability
