#!/usr/bin/env python3
"""
    Accuracy
"""
import tensorflow.compat.v1 as tf


def calculate_accuracy(y, y_pred):
    """
    Method:
        calculates the accuracy of a prediction.

    Parameters:
        @y (float32): placeholder for the labels of the input data
        @y_pred (tensor): the network's predictions.

    Returns:
        a tensor containing the decimal accuracy of the prediction
    """
    # https://jaredwinick.github.io/what_is_tf_keras/

    # For each prediction, if the index with the largest value
    # matches the target value, then the prediction was correct.
    max_correct_predictions_index = tf.math.argmax(y, axis=1)
    max_output_predictions_index = tf.math.argmax(y_pred, axis=1)
    compare_data = tf.math.equal(max_output_predictions_index,
                                 max_correct_predictions_index)

    # Casts compare_data to to be float.
    compare_data_float = tf.dtypes.cast(compare_data, "float")
    # extract accuracy
    accuracy = tf.math.reduce_mean(compare_data_float)

    return accuracy
