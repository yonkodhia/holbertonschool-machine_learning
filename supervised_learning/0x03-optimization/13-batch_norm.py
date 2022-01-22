#!/usr/bin/env python3
"""13-batch_norm.py"""

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """ normalizes an unactivated output of a neural network using batch"""
    ml = np.mean(Z, axis=0)
    rs = np.mean((Z - ml) ** 2, axis=0)
    a = (Z - ml) / np.sqrt(rs + epsilon)
    return gamma * a + beta
