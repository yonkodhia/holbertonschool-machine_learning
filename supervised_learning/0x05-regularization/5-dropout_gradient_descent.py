#!/usr/bin/env python3
""" dropout_gradient_decesnt.py"""

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """ updates the weights and biases of a neural network"""

    m = Y.shape[1]
    lbs = weights.copy()

    for i in range(L - 1, -1, -1):
        prev_A = cache['A' + str(i)]
        W = weights['W' + str(i + 1)]
        b = weights['b' + str(i + 1)]
        A = cache['A' + str(i + 1)]
        if i == L - 1:
            rs = A - Y
        else:
            D = cache['D' + str(i + 1)]
            rs = np.matmul(lbs['W' + str(i + 2)].T, rs) * \
                (1 - (A * A)) * D / keep_prob
        rf = np.matmul(rs, prev_A.T) / m
        rt = np.sum(rs, axis=1, keepdims=True) / m
        weights['W' + str(i + 1)] = weights["W" + str(i + 1)] - (alpha * rf)
        weights['b' + str(i + 1)] = weights["b" + str(i + 1)] - (alpha * rt)
