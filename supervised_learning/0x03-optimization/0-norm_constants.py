#!/usr/bin/env python3
"""
0-norm_constants
"""

import numpy as np


def normalization_constants(X):
    """calc the normalization"""
    m = np.mean(X, axis=0)
    s = np.std(X, axis=0)

    return m, s
