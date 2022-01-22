#!/usr/bin/env python3
"""
8-RMSProp
"""

import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """create RMSProp operation """

    train = tf.train.RMSPropOptimizer(learning_rate=alpha,
                                      decay=beta2,
                                      epsilon=epsilon)

    operation = train.minimize(loss)
    return operation
