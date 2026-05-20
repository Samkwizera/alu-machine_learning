#!/usr/bin/env python3
"""Create the forward propagation graph for a neural network."""

create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """Create the forward propagation graph.

    Args:
        x: Placeholder for the input data.
        layer_sizes: Number of nodes in each layer.
        activations: Activation functions for each layer.

    Returns:
        The network prediction tensor.
    """
    output = x
    for size, activation in zip(layer_sizes, activations):
        output = create_layer(output, size, activation)
    return output
