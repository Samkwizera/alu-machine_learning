#!/usr/bin/env python3
"""
Module `101-the_whole_barn`
"""


def add_matrices(mat1, mat2):
    """
    Adds two matrices element-wise.
    """
    if type(mat1) != type(mat2):
        return None

    if isinstance(mat1, (int, float)):
        return mat1 + mat2

    if len(mat1) != len(mat2):
        return None

    result = []
    for sub_mat1, sub_mat2 in zip(mat1, mat2):
        summed = add_matrices(sub_mat1, sub_mat2)
        if summed is None:
            return None
        result.append(summed)

    return result
