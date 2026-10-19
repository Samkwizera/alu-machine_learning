#!/usr/bin/env python3
"""Baum-Welch training for a hidden Markov model."""

import numpy as np


def _valid_model(Observations, Transition, Emission, Initial, iterations):
    """Validate Baum-Welch inputs."""
    if not isinstance(Observations, np.ndarray) or Observations.ndim != 1:
        return False
    if Observations.size < 2 or not np.issubdtype(Observations.dtype,
                                                 np.integer):
        return False
    if not isinstance(Transition, np.ndarray) or Transition.ndim != 2:
        return False
    if not isinstance(Emission, np.ndarray) or Emission.ndim != 2:
        return False
    if not isinstance(Initial, np.ndarray) or Initial.ndim != 2:
        return False
    if not isinstance(iterations, int) or isinstance(iterations, bool):
        return False
    if iterations < 1:
        return False
    states, outputs = Emission.shape
    if states == 0 or outputs == 0 or Transition.shape != (states, states):
        return False
    if Initial.shape != (states, 1):
        return False
    arrays = (Transition, Emission, Initial)
    if any(not np.all(np.isfinite(array)) for array in arrays):
        return False
    if any(np.any(array < 0) for array in arrays):
        return False
    if (not np.allclose(Transition.sum(axis=1), 1) or
            not np.allclose(Emission.sum(axis=1), 1) or
            not np.isclose(Initial.sum(), 1)):
        return False
    return bool(np.all((Observations >= 0) & (Observations < outputs)))


def baum_welch(Observations, Transition, Emission, Initial,
               iterations=1000):
    """Estimate transition and emission probabilities using EM."""
    if not _valid_model(Observations, Transition, Emission,
                        Initial, iterations):
        return None, None

    transition = Transition.astype(float, copy=True)
    emission = Emission.astype(float, copy=True)
    initial = Initial[:, 0].astype(float, copy=False)
    states = transition.shape[0]
    length = Observations.size

    for _ in range(iterations):
        alpha = np.zeros((states, length))
        scales = np.zeros(length)
        alpha[:, 0] = initial * emission[:, Observations[0]]
        scales[0] = alpha[:, 0].sum()
        if scales[0] == 0:
            return None, None
        alpha[:, 0] /= scales[0]
        for step in range(1, length):
            alpha[:, step] = (alpha[:, step - 1] @ transition) * \
                emission[:, Observations[step]]
            scales[step] = alpha[:, step].sum()
            if scales[step] == 0:
                return None, None
            alpha[:, step] /= scales[step]

        beta = np.ones((states, length))
        for step in range(length - 2, -1, -1):
            future = emission[:, Observations[step + 1]] * \
                beta[:, step + 1]
            beta[:, step] = transition @ future / scales[step + 1]

        gamma = alpha * beta
        gamma_sums = gamma.sum(axis=0, keepdims=True)
        if np.any(gamma_sums == 0):
            return None, None
        gamma /= gamma_sums

        xi_sum = np.zeros_like(transition)
        for step in range(length - 1):
            xi = (alpha[:, step, None] * transition *
                  (emission[:, Observations[step + 1]] *
                   beta[:, step + 1])[None, :])
            total = xi.sum()
            if total == 0:
                return None, None
            xi_sum += xi / total

        transition_denominator = gamma[:, :-1].sum(axis=1)
        emission_denominator = gamma.sum(axis=1)
        if (np.any(transition_denominator == 0) or
                np.any(emission_denominator == 0)):
            return None, None
        transition = xi_sum / transition_denominator[:, None]

        new_emission = np.zeros_like(emission)
        for output in range(emission.shape[1]):
            new_emission[:, output] = gamma[
                :, Observations == output].sum(axis=1)
        emission = new_emission / emission_denominator[:, None]

    return transition, emission
