#!/usr/bin/env python3
"""Calculate the state distribution of a Markov chain."""

import numpy as np


def markov_chain(P, s, t=1):
    """Return the state probabilities after ``t`` transitions."""
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return None
    if P.shape[0] == 0 or P.shape[0] != P.shape[1]:
        return None
    if not isinstance(s, np.ndarray) or s.shape != (1, P.shape[0]):
        return None
    if not isinstance(t, int) or isinstance(t, bool) or t < 0:
        return None
    if (not np.all(np.isfinite(P)) or not np.all(np.isfinite(s)) or
            np.any(P < 0) or np.any(s < 0)):
        return None
    if (not np.allclose(P.sum(axis=1), 1) or
            not np.isclose(s.sum(), 1)):
        return None

    return np.matmul(s, np.linalg.matrix_power(P, t))
