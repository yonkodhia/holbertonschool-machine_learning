#!/usr/bin/env python3
"""Deep neural network.py"""

import numpy as np


class DeepNeuralNetwork:
    """ neural network"""

    def __init__(self, nx, layers):
        """neural netwark"""

        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for law in range(self.__L):
            if not isinstance(layers[law], int) or layers[law] <= 0:
                raise TypeError("layers must be a list of positive integers")
            if law == 0:
                rs = (np.random.randn(layers[law], nx)
                      * np.sqrt(2 / nx))
                self.__weights["W{}".format(law + 1)] = rs
            else:
                rs = (np.random.randn(layers[law], layers[law - 1])
                      * np.sqrt(2 / layers[law - 1]))
                self.__weights["W{}".format(law + 1)] = rs
            self.__weights["b{}".format(law + 1)] = np.zeros((layers[law], 1))

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights
