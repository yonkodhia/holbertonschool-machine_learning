#!/usr/bin/env python3
"""keras evalute model"""


import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """function to test and evaluate a model"""
    return network.evaluate(data, labels, verbose=verbose)
