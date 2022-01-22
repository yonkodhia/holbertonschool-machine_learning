#!/usr/bin/env python3
""" the projection_block"""

import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """   function the project block """
    F11, F3, F12 = filters

    initializer = K.initializers.he_normal(seed=None)

    my_layer = K.layers.Conv2D(filters=F11,
                               kernel_size=(1, 1),
                               strides=(s, s),
                               padding='same',
                               kernel_initializer=initializer,
                               )(A_prev)

    my_layer = K.layers.BatchNormalization(axis=3)(my_layer)
    my_layer = K.layers.Activation('relu')(my_layer)

    my_layer = K.layers.Conv2D(filters=F3,
                               kernel_size=(3, 3),
                               padding='same',
                               kernel_initializer=initializer,
                               )(my_layer)

    my_layer = K.layers.BatchNormalization(axis=3)(my_layer)
    my_layer = K.layers.Activation('relu')(my_layer)

    my_layer = K.layers.Conv2D(filters=F12,
                               kernel_size=(1, 1),
                               padding='same',
                               kernel_initializer=initializer,
                               )(my_layer)

    my_layer = K.layers.BatchNormalization(axis=3)(my_layer)

    short_c = K.layers.Conv2D(filters=F12,
                              kernel_size=(1, 1),
                              strides=(s, s),
                              padding='same',
                              kernel_initializer=initializer,
                              )(A_prev)

    short_c = K.layers.BatchNormalization(axis=3)(short_c)

    output = K.layers.Add()([my_layer, short_c])

    output = K.layers.Activation('relu')(output)

    return output
