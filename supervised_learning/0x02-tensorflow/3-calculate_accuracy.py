#!/usr/bin/env python3


""" calculate accuracy"""
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """ accuracy function"""
    equality = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
    return accuracy
