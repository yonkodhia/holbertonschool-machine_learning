#!/usr/bin/env python3
"""input.py"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """build model function"""
    inputs = K.Input(shape=(nx,))
    L2 = K.regularizers.l2(lambtha)
    a = len(layers)
    for i in range(a):
        if i == 0:
            output = K.layers.Dense(layers[i],
                                    activation=activations[i],
                                    kernel_regularizer=L2)(inputs)
        else:
            dropout = K.layers.Dropout(1 - keep_prob)(output)
            output = K.layers.Dense(layers[i],
                                    activation=activations[i],
                                    kernel_regularizer=L2)(dropout)
    return K.models.Model(inputs=inputs, outputs=output)
