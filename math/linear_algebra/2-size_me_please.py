#!/usr/bin/env python3
"""
Module for 2-size_me_please
"""


def matrix_shape(matrix):
    """
    A function that returns the shape of a matrix as a list of integers.

    :param matrix: A matrix (a nested list of lists)
    :return: List of integers representing the shape of the matrix
    """

    shape = []

    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0]

    return shape
