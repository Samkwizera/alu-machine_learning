#!/usr/bin/env python3
"""
Module - 100-slice_like_a_ninja
"""


def np_slice(matrix, axes={}):
    """
    Slices a matrix along specific axes.
    """
    slices = [slice(None)] * \
        len(matrix.shape)  # Create a default slice for all axes
    for axis, slice_tuple in axes.items():
        # Replace the slice for the specified axis
        slices[axis] = slice(*slice_tuple)
    return matrix[tuple(slices)]
