#!/usr/bin/env python3
"""Save and load  a model weights"""


import tensorflow.keras as K


def save_weights(network, filename, format='h5'):
    """save weights"""
    network.save_weights(filename, format)


def load_weights(network, filename):
    """load weights"""
    return network.load_weights(filename)
