#!/usr/bin/env python3

"""function that return two holders"""

import tensorflow as tf


def create_placeholders(nx, classes):
    """holders(x;y)"""
    x = tf.placeholder(float, shape=[None, nx], name='x')
    y = tf.placeholder(float, shape=[None, classes], name='y')
    return x, y
