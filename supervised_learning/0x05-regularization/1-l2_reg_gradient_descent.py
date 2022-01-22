#!/usr/bin/env python3
"""l1_reg_gradient_descent"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """ updates the weights and biases of a neural network"""

    a = Y.shape[1]
    no = cache['A' + str(L)]
    key = no - Y
    for i in range(L, 0, -1):
        bs = 'b' + str(i)
        Bt = 'W' + str(i)
        shine = weights[Bt]
        line = cache['A' + str(i - 1)]
        fky = weights[Bt]
        hot = (1 / a) * (np.sum(key, axis=1, keepdims=True))
        cold = (1 / a) * np.matmul(key, line.T) + ((lambtha / a) * shine)
        key = np.matmul(fky.T, key) * (1 - line * line)
        weights[Bt] = weights[Bt] - alpha * cold
        weights[bs] = weights[bs] - alpha * hot
