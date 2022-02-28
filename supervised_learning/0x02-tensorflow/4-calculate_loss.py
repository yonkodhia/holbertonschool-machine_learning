#!/usr/bin/env python3
"""
    Loss
"""
import tensorflow.compat.v1 as tf


def calculate_loss(y, y_pred):
    """
    Method:
        calculates the softmax cross-entropy loss of a prediction.

    Parameters:
        @y (float32): placeholder for the labels of the input data
        @y_pred (tensor): the network's predictions.

    Returns:
        tensor containing the loss of the prediction.
    """
    # softmax: an activation function! outputs the probability
    # for each class and these probabilities will sum up to one.
    # Cross Entropy loss is just the sum
    # of the negative logarithm of the probabilities
    # https://www.tensorflow.org/api_docs/python/tf/compat/v1/losses/softmax_cross_entropy
    return tf.losses.softmax_cross_entropy(onehot_labels=y, logits=y_pred)
