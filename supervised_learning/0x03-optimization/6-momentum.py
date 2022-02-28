#!/usr/bin/env python3
"""Optimization module"""
import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """Creates the training operation for a neural network in tensorflow using
    the gradient descent with momentum optimization algorithm

    Args:
        loss (tf.Tensor): Is the loss of the network
        alpha (float): Is the learning rate
        beta1 (float): Is the momentum weight

    Returns:
        tf.operation: Returns the momentum optimization operation

    """
    momentum = tf.train.MomentumOptimizer(alpha, beta1)
    return momentum.minimize(loss)
