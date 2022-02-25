#!/usr/bin/env python3
"""Save and load a model config"""


import tensorflow.keras as K


def save_config(network, filename):
    """Save a config"""
    with open(filename, 'w+') as file:
        file.write(network.to_json())


def load_config(filename):
    """Load a config"""
    with open(filename, 'r') as file:
        return K.models.model_from_json(file.read())
