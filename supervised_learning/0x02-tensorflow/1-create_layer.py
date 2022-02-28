#!/usr/bin/env python3
"""Tensorflow module"""
import tensorflow as tf


def create_layer(prev, n, activation):
    """Creates a tensorflow layer

    Args:
        prev (tf.tensor): Is tensor output of the previous layer
        n (int): Is the number of nodes in the layer.
        activation: Is the activation function that the layer should use

    Returns:
        tf.tensor: Returns the tensor output of the layer

    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    model = tf.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=init,
        name="layer",
    )
    output = model(prev)
    return output
