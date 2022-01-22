#!/usr/bin/env python3
"""
6-momentum
"""

import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """function that create momentum"""

    return tf.train.MomentumOptimizer(alpha, beta1).minimize(loss)
