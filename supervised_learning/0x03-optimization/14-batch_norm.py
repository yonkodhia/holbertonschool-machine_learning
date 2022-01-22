#!/usr/bin/env python3
"""14-batch_norm.py"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """creates a batch normalization layer for a neural network"""
    kernel = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    rs = tf.layers.dense(prev, units=n, activation=None,
                         kernel_initializer=kernel, name="layer",
                         reuse=tf.AUTO_REUSE)

    a, b = tf.nn.moments(rs, axes=[0])
    gamma = tf.Variable(tf.constant(1.0, shape=[n]),
                        name="gamma", trainable=True)
    beta = tf.Variable(tf.constant(0.0, shape=[n]),
                       name="beta", trainable=True)
    rushs = tf.nn.batch_normalization(rs, mean=a, variance=b,
                                      offset=beta, scale=gamma,
                                      variance_epsilon=1e-8)
    return activation(rushs)
