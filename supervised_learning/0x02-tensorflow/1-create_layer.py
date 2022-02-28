#!/usr/bin/env python3
"""
    Creating a tensorflow Layer.
"""
import tensorflow.compat.v1 as tf


def create_layer(prev, n, activation):
    """
    Method:
        Creating a layer.

    Parameters:
        @prev: is the tensor output of the previous layer
        @n: is the number of nodes in the layer to create
        @activation: is the activation function that the layer
        should use

    Returns:
        the tensor output of the layer
    """
    # https://keras.io/api/layers/core_layers/dense/

    init_weights = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    # https://www.tensorflow.org/guide/keras/sequential_model

    layer = tf.keras.layers.Dense(n,
                                  activation=activation,
                                  kernel_initializer=init_weights,
                                  name="layer")

    Input_data = prev
    return layer(Input_data)
