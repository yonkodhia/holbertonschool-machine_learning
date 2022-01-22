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

    def cost(self, Y, A):
        """
        cost function
        :param A:
        :param Y:
        :return: the cost
        """
        c = - (np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)))
        c /= Y.shape[1]
        return c

    def evaluate(self, X, Y):
        """evaluate the neuron's pred"""
        A = self.forward_prop(X)
        c = self.cost(Y, A)
        return np.where(A >= 0.5, 1, 0), c

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """calculate one pass of gradient"""
        m = Y.shape[1]
        B = np.transpose(A)
        x = np.dot((A - Y), X.T) / m
        y = (np.sum(A - Y)) / m
        self.__W += - alpha * x
        self.__b += - alpha * y
