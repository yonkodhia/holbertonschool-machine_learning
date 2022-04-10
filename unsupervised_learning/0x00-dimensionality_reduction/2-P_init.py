#!/usr/bin/env python3
"""Init variables needed to calculate P"""


import numpy as np


def P_init(X, perplexity):
    """function P initialisation"""
    D = np.ndarray((X.shape[0], X.shape[0]))
    for point in range(X.shape[0]):
        D[point] = np.square((X - X[point])).sum(axis=1).T
    H = np.log2(perplexity)
    return D, np.zeros((X.shape[0], X.shape[0])), np.ones((X.shape[0], 1)), H
