#!/usr/bin/env python3
"""10-Adam.py"""

import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """create function that update training operation"""

    return tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon).minimize(loss)
