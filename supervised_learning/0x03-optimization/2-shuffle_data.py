#!/usr/bin/env python3
"""2-Shuffle_data"""

import numpy as np


def shuffle_data(X, Y):
    """Shuffle x & y"""
    m = X.shape[0]
    sp = np.random.permutation(m)

    return X[sp], Y[sp]
