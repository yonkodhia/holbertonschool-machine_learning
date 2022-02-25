#!/usr/bin/env python3
"""
add class
"""

import matplotlib.pyplot as plt
import numpy as np


class Neuron:
    """neuron class"""

    def __init__(self, nx):
        """constructor"""
        if not type(nx) is int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')

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
        """
        forward propagation
        """
        i = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-i))
        return self.__A

    def cost(self, Y, A):
        """
        cost function
        :param Y:
        :param A:
        :return:
        """
        cost = - (np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)))
        cost /= Y.shape[1]
        return cost

    def evaluate(self, X, Y):
        """
        function to evaluate the neuron’s predictions
        :param X:
        :param Y:
        :return:
        """
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        return np.where(A >= 0.5, 1, 0), cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        function Calculates one pass of gradient descent on the neuron
        :param X:
        :param Y:
        :param A:
        :param alpha:
        :return:
        """
        a = X.shape[1]
        b = A - Y
        c = np.sum(X * b, axis=1)
        j = np.sum(b)
        c /= a
        j /= a
        self.__W = self.__W - (alpha * c)
        self.__b = self.__b - (alpha * j)

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
        function to train neurone
        :param step:
        :param graph:
        :param verbose:
        :param self:
        :param X:
        :param Y:
        :param iterations:
        :param alpha:
        :return:
        """
        co = []
        it = []
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if (verbose is True and graph is False) or \
                (verbose is False and graph is True):
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        for i in range(0, iterations + 1):
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A, alpha)
            if i == 0 or i % step == 0 or i == iterations:
                cost = self.cost(Y, self.__A)
                co.append(i)
                it.append(cost)
                if verbose is True:
                    print("Cost after {} iterations: {}".format(i, cost))
        if graph is True:
            plt.plot(co, it)
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()
        return self.evaluate(X, Y)
