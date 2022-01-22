#!/usr/bin/env python3
"""l2_reg_create_layer"""

import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """layer that includes L2 regularization: """

    lima = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    regularizers = tf.contrib.layers.l2_regularizer(lambtha)
    leyer = tf.layers.Dense(units=n, activation=activation,
                            use_bias=True,
                            kernel_initializer=lima,
                            kernel_regularizer=regularizers,
                            bias_regularizer=None)
    return leyer(prev)
