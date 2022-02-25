#!/usr/bin/env python3
"""
module to create tensorflow placeholders
"""


import tensorflow as tf


def create_placeholders(nx, classes):
    """
    create placeholders
    """
    x = tf.placeholder(name='x', dtype=tf.float32, shape=(None, nx))
    y = tf.placeholder(name='y', dtype=tf.float32, shape=(None, classes))
    return x, y
