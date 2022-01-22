#!/usr/bin/env python3
""" 1-create_layer"""
import tensorflow as tf


def create_layer(prev, n, activation):
    """function to create a layer"""
    pid = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    flake = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=deb, kernel_constraint=None,
                            name='layer')

    return flake(prev)
