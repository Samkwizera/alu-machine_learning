#!/usr/bin/env python3
"""Update a learning rate using inverse time decay."""


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """Return the stepwise inverse-time decayed learning rate."""
    return alpha / (1 + (decay_rate * (global_step // decay_step)))
