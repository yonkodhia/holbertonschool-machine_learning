#!/usr/bin/env python3

"""
CLASS deep neural network
"""

import numpy as np


class DeepNeuralNetwork:
    """class deeplearning"""

    def __init__(self, nx, layers):
        """costructor"""
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if type(layers) is not list or not layers:
            raise TypeError('layers must be a list of positive integers')
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for i in range(len(layers)):
            if layers[i] <= 0:
                raise TypeError("layers must be a list of positive integers")
            if i == 0:
                v = np.random.randn(layers[i], nx) * np.sqrt(2 / nx)
                self.__weights['W' + str(i + 1)] = v
            else:
                self.__weights['W' + str(i + 1)] = \
                    np.random.randn(layers[i], layers[i - 1]) * \
                    np.sqrt(2 / layers[i - 1])
            self.__weights['b' + str(i + 1)] = np.zeros((layers[i], 1))

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    def forward_prop(self, X):
        """ Calculates the forward propagation"""
        self.__cache["A0"] = X
        for i in range(self.__L):
            W = self.__weights["W" + str(i + 1)]
            A = self.__cache["A" + str(i)]
            b = self.__weights["b" + str(i + 1)]
            Z = np.matmul(W, A) + b
            self.__cache["A" + str(i + 1)] = 1 / (1 + np.exp(-Z))
        return self.__cache["A" + str(i + 1)], self.__cache

    def cost(self, Y, A):
        """ calculates the cost"""
        cost = - (np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)))
        cost /= Y.shape[1]
        return cost

    def evaluate(self, X, Y):
        """ Evaluates the neuron’s predictions"""
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        return np.where(A >= 0.5, 1, 0), cost
