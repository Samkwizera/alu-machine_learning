#!/usr/bin/env python3
"""
Module 5-across_the_planes
"""


def add_matrices2D(mat1, mat2):
    """
    Module-5
    """
    if len(mat1) != len(mat2):
        return None
    for row1, row2 in zip(mat1, mat2):
        if len(row1) != len(row2):
            return None
    result = []

    for i in range(len(mat1)):
        row_sum = [mat1[i][j] + mat2[i][j] for j in range(len(mat1[i]))]
        result.append(row_sum)
    return result
