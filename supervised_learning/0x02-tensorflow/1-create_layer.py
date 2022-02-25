#!/usr/bin/env python3


"""create layers"""
import tensorflow as tf


def create_layer(prev, n, activation):
    """
    prev
    n
    activation
    """
    x = tf.contrib.layers.variance_scaling_initializer(mode='FAN_AVG')
    layer = tf.layers.Dense(units=n,
                            kernel_initializer=x,
                            activation=activation,
                            name='layer')
    y = layer(prev)
    return y
