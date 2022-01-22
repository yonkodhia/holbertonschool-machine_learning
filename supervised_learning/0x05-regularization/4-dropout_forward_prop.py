#!/usr/bin/env python3
""" dropout_forward_prop.py"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """conducts forward propagation using Dropout"""

    m = X.shape[1]
    cache = {'A0': X}
    for i in range(L):
        v = 'A' + str(i + 1)
        A_prev = 'A' + str(i)
        W = weights['W' + str(i + 1)]
        b = weights['b' + str(i + 1)]
        Z = np.matmul(W, cache[A_prev]) + b
        if i == L - 1:
            t = np.exp(Z)
            cache[v] = t / np.sum(t, axis=0, keepdims=True)
        else:
            cache[v] = 2 / (1 + np.exp(-2 * Z)) - 1
            d = np.random.rand(
                cache[v].shape[0],
                cache[v].shape[1]) < keep_prob
            cache['D' + str(i + 1)] = np.where(d, 1, 0)
            cache[v] *= d
            cache[v] /= keep_prob
    return cache
