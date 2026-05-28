#!/usr/bin/env python3
"""Calculate the weighted moving average of a data set."""


def moving_average(data, beta):
    """Return the bias-corrected weighted moving average of data."""
    averages = []
    weighted_average = 0

    for i, value in enumerate(data, 1):
        weighted_average = (beta * weighted_average) + ((1 - beta) * value)
        averages.append(weighted_average / (1 - (beta ** i)))

    return averages
