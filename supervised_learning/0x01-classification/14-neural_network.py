#!/usr/bin/env python3
"""
function neural network
"""

import numpy as np


class NeuralNetwork:
    """neural network class """
    def __init__(self, nx, nodes):
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if not isinstance(nodes, int):
            raise TypeError('nodes must be an integer')
        if nodes < 1:
            raise ValueError('nodes must be a positive integer')
        self.nx = nx
        self.nodes = nodes
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        return self.__W1

    @property
    def W2(self):
        return self.__W2

    @property
    def b1(self):
        return self.__b1

    @property
    def b2(self):
        return self.__b2

    @property
    def A1(self):
        return self.__A1

    @property
    def A2(self):
        return self.__A2

    def forward_prop(self, X):
        """ Calculates the  prop"""
        a = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-a))
        b = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-b))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """Calculates the cost using logisitic"""
        c = - (np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)))
        c /= Y.shape[1]
        return c

    def evaluate(self, X, Y):
        """calc the evaluate neural"""
        self.forward_prop(X)
        pred = np.where(self.A2 >= 0.5, 1, 0)
        cost = self.cost(Y, self.A2)
        return pred, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """ Calculate gradient decent """
        m = Y.shape[1]
        dz2 = A2 - Y
        dz1 = np.dot(self.W2.T, dz2) * A1 * (1 - A1)
        self.__W2 += - alpha * np.dot(dz2, A1.T) / m
        self.__b2 += - alpha * np.sum(dz2, axis=1) / m
        self.__W1 += - alpha * np.dot(dz1, X.T) / m
        self.__b1 += - alpha * np.sum(dz1, axis=1, keepdims=True) / m

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """function to train network"""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        for i in range(iterations):
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A1, self.__A2, alpha)

        return self.evaluate(X, Y)
