#!/usr/bin/env python3
"""Positional encoding for a transformer."""

import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    max_seq_len is the maximum sequence length
    dm is the model depth

    Returns: a numpy.ndarray of shape (max_seq_len, dm)
    """
    PE = np.zeros((max_seq_len, dm))
    position = np.arange(max_seq_len)[:, np.newaxis]
    index = np.arange(0, dm, 2)
    div_term = np.power(10000, index / dm)

    PE[:, 0::2] = np.sin(position / div_term)
    PE[:, 1::2] = np.cos(position / div_term)
    return PE
