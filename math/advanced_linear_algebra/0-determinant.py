#!/usr/bin/env python3
"""
Module for 0-determinant
"""


def determinant(matrix):
    """Calculates the determinant of a matrix."""
    if not isinstance(matrix, list) or not all(
            isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # Handle 0x0 matrix
    if matrix == [[]]:
        return 1

    n = len(matrix)

    # Ensure it's a square matrix
    if not all(len(row) == n for row in matrix):
        raise ValueError("matrix must be a square matrix")

    # Base case for 1x1 matrix
    if n == 1:
        return matrix[0][0]

    # Base case for 2x2 matrix
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    # Recursive case for larger matrices
    det = 0
    for i in range(n):
        sub_matrix = [row[:i] + row[i + 1:]
                      for row in matrix[1:]]  # Minor of matrix[0][i]
        det += ((-1) ** i) * matrix[0][i] * determinant(sub_matrix)

    return det
