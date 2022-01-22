#!/usr/bin/env python3
"""file inception block"""
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """function inception_block"""
    F1, F3R, F3, F5R, F5, FPP = filters

    my_layer = K.layers.Conv2D(
        filters=F1,
        kernel_size=1,
        activation='relu',
        padding='same',
    )(A_prev)
    my_layer1 = K.layers.Conv2D(
        filters=F3R,
        kernel_size=1,
        activation='relu',
        padding='same',
    )(A_prev)
    my_layer0 = K.layers.Conv2D(
        filters=F3,
        kernel_size=3,
        activation='relu',
        padding='same',
    )(my_layer1)
    my_layer2 = K.layers.Conv2D(
        filters=F5R,
        kernel_size=1,
        activation='relu',
        padding='same',
    )(A_prev)
    my_layer3 = K.layers.Conv2D(
        filters=F5,
        kernel_size=5,
        activation='relu',
        padding='same',
    )(my_layer2)
    my_layer5 = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=1,
        padding='same'
    )(A_prev)
    my_layer4 = K.layers.Conv2D(
        filters=FPP,
        kernel_size=1,
        activation='relu',
        padding='same',
    )(my_layer5)
    return K.layers.Concatenate()([my_layer, my_layer0, my_layer3, my_layer4])
