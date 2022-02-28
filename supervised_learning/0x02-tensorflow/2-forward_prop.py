#!/usr/bin/env python3
"""Tensorflow module"""
import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """Creates the forward propagation graph for the neural netword

    Args:
        x (tf.placeholder): Is the input data
        layer_sizes (list): Is containing the number of nodes in each layer
        of the network
        activations (list): Is containing the activation functions for each
        layer of the network.

    Returns:
        tf.Tensor: Returns the prediction of the network in tensor form

    """
    prediction = x
    for layer, activation in zip(layer_sizes, activations):
        prediction = create_layer(prediction, layer, activation)
    return prediction
