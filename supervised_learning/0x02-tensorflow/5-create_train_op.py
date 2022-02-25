#!/usr/bin/env python3


"""train function"""
import tensorflow as tf


def create_train_op(loss, alpha):
    """train"""
    return tf.train.GradientDescentOptimizer(alpha).minimize(loss)
