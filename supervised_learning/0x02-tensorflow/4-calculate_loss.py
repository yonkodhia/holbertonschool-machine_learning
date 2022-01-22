#!/usr/bin/env python3
"""
Contains a function to calculate the softmax cross-entropy loss of a prediction
"""

import tensorflow as tf


def calculate_loss(y, y_pred):
    """calc the loss"""
    lost = tf.losses.softmax_cross_entropy(y, y_pred)
    return lost
