#!/usr/bin/env python3
"""Tensorflow module"""
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """Calculates the accuracy of a prediction

    Args:
        y (tf.placeholder): Is the placeholder for the input data
        y_pred (tf.Tensor): Is a tensor containing the network's prediction

    Returns:
        tf.Tensor: Containing the decimal accuracy of the predcition

    """
    truth_max = tf.argmax(y, 1)
    pred_max = tf.argmax(y_pred, 1)
    difference = tf.equal(truth_max, pred_max)
    accuracy = tf.reduce_mean(tf.cast(difference, "float"))
    return accuracy
