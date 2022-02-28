#!/usr/bin/env python3
"""
    Train_Op
"""
import tensorflow.compat.v1 as tf


def create_train_op(loss, alpha):
    """
    Method:
        creates the training operation for the network.

    Parameters:
        @loss:  the loss of the network's prediction.
        @alpha: the learning rate.

    Returns:
        an operation that trains the network using
        gradient descent.
    """
    # https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/Optimizer

    # Create an optimizer.
    opt = tf.train.GradientDescentOptimizer(learning_rate=alpha)

    return opt.minimize(loss)
