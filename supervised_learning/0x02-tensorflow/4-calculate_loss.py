#!/usr/bin/env python3
"""Tensorflow module"""
import tensorflow as tf


def calculate_loss(y, y_pred):
    """Calculates the softmax cross-entropy loss of a prediction

    Args:
        y (tf.placeholder): the labels of the input data
        y_pred (tf.Tensor): Is a tensor containing the network's prediction

    Returns:
        tf.Tensor: Containing the loss of the prediction

    """
    return tf.losses.softmax_cross_entropy(y, y_pred)
