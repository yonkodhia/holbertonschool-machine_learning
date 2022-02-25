#!/usr/bin/env python3
"""Save and load a model with keras"""


import tensorflow.keras as K


def save_model(network, filename):
    """save model Network to filename"""
    network.save(filename)


def load_model(filename):
    """load model with keras"""
    return K.models.load_model(filename)
