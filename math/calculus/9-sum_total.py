#!/usr/bin/env python3
""" Summation Function """


def summation_i_squared(n):
    """
    Summation of i^2 from i = 1 to n

    Utilizes the formula:
    sigma i^2 = n(n + 1)(2n + 1) / 6
    """
    if type(n) is not int or n < 1:
        return None
    sigma_sum = (n * (n + 1) * ((2 * n) + 1)) / 6
    return int(sigma_sum)
