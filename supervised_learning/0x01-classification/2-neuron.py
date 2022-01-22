#!/usr/bin/env python3
"""
function 1-neruon
"""

import numpy as np


class Neuron:
    """ class neuro"""

    def __init__(self, nx):
        """function init"""
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        elif nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A

    def forward_prop(self, X):
        """calc the cost of model usining logisitic"""
        o = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-o))
        return self.__A
