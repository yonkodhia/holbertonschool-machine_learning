#!/usr/bin/env python3
"""dropout_create_layer"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """"creates a layer of a neural network using dropout:"""
    rtfm = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    regularizers = tf.layers.Dropout(keep_prob)
    layer = tf.layers.Dense(units=n,
                            activation=activation,
                            kernel_initializer=rtfm,
                            kernel_regularizer=regularizers,
                            bias_regularizer=None,
                            name='layer')
    return layer(prev)
