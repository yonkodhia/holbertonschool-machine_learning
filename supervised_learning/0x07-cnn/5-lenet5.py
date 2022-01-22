#!/usr/bin/env python3
"""File 5-lenet5"""


import tensorflow.keras as K


def lenet5(X):
    """function that modifie archt of lenet 5"""
    ola = K.layers.Conv2D(6, 5, padding='same',
                          activation='relu',
                          kernel_initializer='he_normal')(X)
    ola = K.layers.MaxPooling2D(2, 2)(ola)
    ola = K.layers.Conv2D(16, 5, padding='valid', activation='relu',
                          kernel_initializer='he_normal')(ola)
    ola = K.layers.MaxPooling2D(2, 2)(ola)
    ola = K.layers.Flatten()(ola)
    ola = K.layers.Dense(120, activation='relu',
                         kernel_initializer='he_normal')(ola)
    ola = K.layers.Dense(84, activation='relu',
                         kernel_initializer='he_normal')(ola)
    ola = K.layers.Dense(10, activation='softmax',
                         kernel_initializer='he_normal')(ola)
    model = K.Model(X, ola)
    model.compile('Adam', metrics=['accuracy'],
                  loss='categorical_crossentropy')
    return model
