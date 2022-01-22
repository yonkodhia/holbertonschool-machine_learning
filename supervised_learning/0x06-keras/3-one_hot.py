#!/usr/bin/env python3
"""one_hot file"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """one hot"""
    HO = K.utils.to_categorical(labels, classes)
    return HO
