#!/usr/bin/env python3
"""Determine whether gradient descent should stop early."""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """Return whether to stop early and the updated patience count."""
    if opt_cost - cost > threshold:
        return False, 0

    count += 1
    return count >= patience, count
