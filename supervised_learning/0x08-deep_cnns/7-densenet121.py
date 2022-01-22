#!/usr/bin/env python3
""" resnet50 function """

import tensorflow.keras as K

dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """function that build densnet arch"""
    initializer = K.initializers.he_normal(seed=None)

    X = K.Input(shape=(224, 224, 3))
    layers = [12, 24, 16]

    my_layer = K.layers.BatchNormalization(axis=3)(X)
    my_layer = K.layers.Activation('relu')(my_layer)

    my_layer = K.layers.Conv2D(filters=2 * growth_rate,
                               kernel_size=(7, 7),
                               strides=(2, 2),
                               padding='same',
                               kernel_initializer=initializer,
                               )(my_layer)

    my_layer = K.layers.MaxPool2D(pool_size=(3, 3),
                                  padding='same',
                                  strides=(2, 2))(my_layer)

    nb_filters = 2 * growth_rate

    my_layer, nb_filters = dense_block(my_layer, nb_filters, growth_rate, 6)

    for layer in layers:
        my_layer, nb_filters = transition_layer(my_layer,
                                                nb_filters,
                                                compression)

        my_layer, nb_filters = dense_block(my_layer,
                                           nb_filters,
                                           growth_rate,
                                           layer)

    my_layer = K.layers.AveragePooling2D(pool_size=(7, 7),
                                         padding='same')(my_layer)

    my_layer = K.layers.Dense(units=1000,
                              activation='softmax',
                              kernel_initializer=initializer,
                              )(my_layer)

    model = K.models.Model(inputs=X, outputs=my_layer)

    return model
