#!/usr/bin/env python3


"""function loss"""
import tensorflow as tf


def calculate_loss(y, y_pred):
    """ loss calculate"""
    return tf.losses.softmax_cross_entropy(y, y_pred)
