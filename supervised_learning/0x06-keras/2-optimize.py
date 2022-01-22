#!/usr/bin/env python3
"""2-optimize"""

import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """function optimize_model"""
    a = K.optimizers.Adam(lr=alpha, beta_1=beta1, beta_2=beta2)
    network.compile(loss='categorical_crossentropy', optimizer=a,
                    metrics=['accuracy'])
    return None
