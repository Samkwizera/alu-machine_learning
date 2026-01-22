#!/usr/bin/env python3
"""
Module 4-line_up
"""


def add_arrays(arr1, arr2):
    """
    Adds two matrices element-wise.
    """
    if len(arr1) != len(arr2):
        return None
    result = []

    for i in range(len(arr1)):
        result.append(arr1[i] + arr2[i])

    return result
