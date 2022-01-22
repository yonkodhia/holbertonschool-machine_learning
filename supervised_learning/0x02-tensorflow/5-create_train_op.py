#!/usr/bin/env python3
"""
create the training operation for the network
"""

import tensorflow as tf


def create_train_op(loss, alpha):
    """function train contain loss and the alpha"""
    calculation = tf.train.GradientDescentOptimizer(alpha).minimize(loss)
    return calculation
