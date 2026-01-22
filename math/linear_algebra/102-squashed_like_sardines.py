#!/usr/bin/env python3
"""
Module `102-squashed_like_sardines`
"""


def cat_matrices(mat1, mat2, axis=0):
    """
    Concatenates two matrices along a specific axis.
    """
    # Check if both inputs are lists
    if not isinstance(mat1, list) or not isinstance(mat2, list):
        return None

    # If axis is 0, check that the inner dimensions match
    if axis == 0:
        if all(isinstance(row, list) for row in mat1) and all(isinstance(row, list) for row in mat2):
            # Ensure all inner dimensions are equal
            if len(mat1[0]) != len(mat2[0]):
                return None
        elif any(isinstance(row, list) for row in mat1) or any(isinstance(row, list) for row in mat2):
            # If one is nested and the other is not, they cannot be concatenated
            return None
        return mat1 + mat2

    # For axis > 0, recurse into the submatrices
    if len(mat1) != len(mat2):
        return None

    result = []
    for sub_mat1, sub_mat2 in zip(mat1, mat2):
        concatenated = cat_matrices(sub_mat1, sub_mat2, axis=axis - 1)
        if concatenated is None:
            return None
        result.append(concatenated)

    return result
