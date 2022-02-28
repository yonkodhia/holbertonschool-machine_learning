#!/usr/bin/env python3
"""Tensorflow module"""
import tensorflow as tf


def create_train_op(loss, alpha):
    """Creates the training operation for the network

    Args:
        loss (tf.Tensor): Is the loss of the network's prediction
        alpha (tf.float): Is the learning rate

    Returns:
        (tf.op): Trains the network using gradient descent

    """
    ops = tf.train.GradientDescentOptimizer(alpha)
    return ops.minimize(loss)
