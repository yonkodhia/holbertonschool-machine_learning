#!/usr/bin/env python3
"""
    Forward Propagation
"""
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Method:
        creates the forward propagation graph
        for the neural network.

    Parameters:
        @x (float32): the placeholder for the input data
        @layer_sizes (list): the number of nodes in each
            layer of the network
        @activations (list): the activation functions for each
            layer of the network

    Returns:
        the prediction of the network in tensor form
    """
    # placeholder: It allows us to create our operations
    # and build our computation graph, without needing the data.

    # the input layer (a placeholder object)
    input_layer = x
    if len(layer_sizes) == len(activations):
        # Create the rest of layers (network)
        for i in range(len(layer_sizes)):
            # to create an output layer we need the previous layer!
            output_layer = create_layer(input_layer,
                                        layer_sizes[i],
                                        activations[i])
            input_layer = output_layer

        return output_layer
    return input_layer
