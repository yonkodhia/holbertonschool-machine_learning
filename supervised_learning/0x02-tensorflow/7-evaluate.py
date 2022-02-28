#!/usr/bin/env python3
"""Tensorflow module"""
import tensorflow as tf


def evaluate(X, Y, save_path):
    """Evaluates the output of a neural network

    Args:
        X (numpy.ndarray): The input data
        Y (numpy.ndarray): Is the one-hot labels for X
        save_path (str): Is the path location of the model.

    Return:
        float: prediction, accuracy, loss

    """
    with tf.Session() as session:
        saver = tf.train.import_meta_graph("{}.meta".format(save_path))
        saver.restore(session, save_path)
        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        y_pred = tf.get_collection("y_pred")[0]
        accuracy = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]
        y_pred, accuracy, loss = session.run(
            [y_pred, accuracy, loss], feed_dict={x: X, y: Y}
        )
    return y_pred, accuracy, loss
