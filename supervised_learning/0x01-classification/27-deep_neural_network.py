#!/usr/bin/env python3
"""Deep Neural Network module"""
import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
    """DeepNeural Network class that defines a neural network performing binary
    classification.

    Attributes:
        nx (int): Is the number of input features.
        layers (list): Is the list contains the lenght of the network layers
        L (int): the number of layers in the neural network
        cache (dict): Holds all the intermediary values of the network
        weights (dict): Holds all the weights and biased of the network.

    Raises:
        TypeError: If nx is not an integer
        ValueError: If nx less than 1
        TypeError: If layers is not a list on positive integer

    """
    def __init__(self, nx, layers):
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if not isinstance(layers, list):
            raise TypeError('layers must be a list of positive integers')
        if not all(map(lambda x: x > 0 and isinstance(x, int), layers)):
            raise TypeError('layers must be a list of positive integers')
        if len(layers) == 0:
            raise TypeError('layers must be a list of positive integers')

        self.nx = nx
        self.layers = layers
        self.__L = len(layers)
        self.__cache = dict()
        self.__weights = dict()
        lst = layers.copy()
        lst.insert(0, self.nx)
        for l in range(1, self.L + 1):
            self.__weights['W' + str(l)] = np.random.randn(
                lst[l], lst[l - 1]) * np.sqrt(2 / lst[l - 1])
            self.__weights['b' + str(l)] = np.zeros((lst[l], 1))

    @property
    def L(self):
        """L getter"""
        return self.__L

    @property
    def cache(self):
        """cache getter"""
        return self.__cache

    @property
    def weights(self):
        """weights getter"""
        return self.__weights

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network

        Args:
            X (numpy.ndarray): Is the input data with shape (nx, m) where
            nx is the number of input features of the neuron and m the number
            of examples.

        Returns:
            (numpy.ndarray): The output of the neural network.

        """
        self.__cache["A0"] = X
        for layer in range(1, self.L + 1):
            z_tmp = np.matmul(self.weights["W" + str(layer)], self.__cache[
                "A" + str(layer - 1)]) + self.weights["b" + str(layer)]
            if layer == self.L:
                t_exp = np.exp(z_tmp)
                A_tmp = t_exp / np.sum(t_exp, axis=0, keepdims=True)
            else:
                A_tmp = 1 / (1 + np.exp((-1) * z_tmp))
            self.__cache["A" + str(layer)] = A_tmp
        return self.cache["A" + str(self.L)], self.cache

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression

        Args:
            Y (numpy.ndarray): Is the correct labels for the input data
            with shape (1, m) where m is the number of example
            A (numpy.ndarray): Is the activated output of the neuron for each
            example.

        Returns:
            float: Returns the cost.

        """
        m = len(Y[0])
        L = Y * np.log(A)
        return (-1/m) * np.sum(L)

    def evaluate(self, X, Y):
        """Evaluates the neural network predictions

        Args:
            X (numpy.ndarray): Is the input data with shape (nx, m) where nx
            is the number of features and m is the number of examples
            Y (numpy.ndarray): Is the correct labels for the input data.

        Returns:
            numpy.ndarray: the predicted labels for each example
            float: the cost of the network.

        """
        self.forward_prop(X)
        A = self.cache["A" + str(self.L)]
        prediction = np.eye(A.shape[0])[np.argmax(A, axis=0)].T
        cost = self.cost(Y, A)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculates one pass of gradient descent on the neural network

        Args:
            Y (numpy.ndarray): Is the correct labels for the input data
            cache (dict): Contains all intermediary values of the network
            alpha (float): Is the learning rate

        """
        m = len(Y[0])
        d_z = self.cache["A" + str(self.L)] - Y
        for l in range(self.L, 0, -1):
            A_prev = self.cache["A" + str(l - 1)]
            d_W = np.matmul(d_z, A_prev.T) / m
            d_b = np.sum(d_z, axis=1, keepdims=True) / m
            d_sig = A_prev * (1 - A_prev)
            d_z = np.matmul(self.weights["W" + str(l)].T, d_z) * d_sig
            self.__weights["W" + str(l)] -= alpha * d_W
            self.__weights["b" + str(l)] -= alpha * d_b

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """Trains the deep neural network

        Args:
            X (numpy.ndarray): Is the input data
            Y (numpy.ndarray): Is the correct labels for each input
            iterations (int): Is the number of iterations to train over
            alpha (flaot): Is the learning rate

        Raises:
            TypeError: If iterations is not an integer
            ValueError: If iterations is not a poistive integer
            TypeError: If alpha is not a float
            ValueError: If alpha is not a positive number

        Returns:
            numpy.ndarray: The evaluation of the training data

        """
        costs = np.array(())
        if not isinstance(iterations, int):
            raise TypeError('iterations must be an integer')
        if iterations < 0:
            raise ValueError('iterations must be a positive integer')
        if not isinstance(alpha, float):
            raise TypeError('alpha must be a float')
        if alpha < 0:
            raise ValueError('alpha must be positive')
        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError('step must be an integer')
            if step not in range(0, iterations + 1):
                raise ValueError('step must be positive and <= iterations')
        for iteration in range(iterations):
            self.forward_prop(X)
            self.gradient_descent(Y, self.cache, alpha)
            if verbose:
                cost = self.cost(Y, self.cache["A" + str(self.L)])
                costs = np.append(costs, cost)
                if iteration % step == 0:
                    print("Cost after {} iterations: {}".format(
                        iteration, cost))
        if graph:
            plt.plot(costs, 'b-')
            plt.title('Training Cost')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.show()
        return self.evaluate(X, Y)

    def save(self, filename):
        """Saves the instance object to a file in a pickle format"""
        if filename[-4:] != ".pkl":
            filename += ".pkl"
        file_obj = open(filename, "wb")
        pickle.dump(self, file_obj)
        file_obj.close()

    @staticmethod
    def load(filename):
        """loads pickled instance object"""
        try:
            file_obj = open(filename, 'rb')
        except Exception:
            return None
        instance_obj = pickle.load(file_obj)
        file_obj.close()
        return instance_obj
