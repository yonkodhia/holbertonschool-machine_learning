#!/usr/bin/env python3
"""keras perdict function"""


import tensorflow.keras as K


def predict(network, data, verbose=False):
    """predict function"""
    return network.predict(data, verbose=verbose)
