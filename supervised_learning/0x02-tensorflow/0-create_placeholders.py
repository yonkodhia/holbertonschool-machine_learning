#!/usr/bin/env python3
"""Tensorflow module"""
import tensorflow as tf


def create_placeholders(nx, classes):
    """Creates two placeholder x and y

    Args:
        nx (int): Is the number of feature columns in our data
        classes (int): Is the number of classes in the classifier

    Returns:
        tf.placeholder: returns a placeholder for the input data to the neural
        network.
        tf.placeholder: returns a placeholder for the one-hot labels for the
        input data

    """
    x = tf.placeholder("float", shape=(None, nx), name="x")
    y = tf.placeholder("float", shape=(None, classes), name="y")
    return x, y
