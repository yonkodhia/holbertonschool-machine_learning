#!/usr/bin/env python3
"""
calculate the accuracy of a prediction
"""

import tensorflow as tf


def calculate_accuracy(y, y_pred):

    """function calc accuracy"""
    correct_predictedd = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accurate = tf.reduce_mean(tf.cast(correct_predictedd, tf.float32))
    return accurate
