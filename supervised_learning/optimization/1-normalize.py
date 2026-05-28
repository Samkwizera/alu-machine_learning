#!/usr/bin/env python3
"""Normalize a matrix using standardization constants."""


def normalize(X, m, s):
    """Return the standardized X matrix."""
    return (X - m) / s
