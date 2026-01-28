#!/usr/bin/env python3
"""
Module for 1-minor
"""


def minor(matrix):
    """
    Calculates the minor matrix of a matrix.
    """

    if not isinstance(matrix, list) or not all(
            isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    n = len(matrix)
    if n == 0 or any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    if n == 1:
        return [[1]]

    def determinant(mat):
        """Helper function to calculate determinant of a matrix."""
        if len(mat) == 1:
            return mat[0][0]
        if len(mat) == 2:
            return mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]
        det = 0
        for j in range(len(mat)):
            submatrix = [row[:j] + row[j + 1:] for row in mat[1:]]
            det += ((-1) ** j) * mat[0][j] * determinant(submatrix)
        return det

    minor_matrix = []
    for i in range(n):
        minor_row = []
        for j in range(n):
            submatrix = [
                row[:j] + row[j + 1:] for row in (matrix[:i] + matrix[i + 1:])
            ]
            minor_row.append(determinant(submatrix))
        minor_matrix.append(minor_row)

    return minor_matrix
