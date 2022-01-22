#!/usr/bin/env python3
"""
9-Adam.py
"""

import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """function that update variables"""

    Ss = (beta1 * v) + ((1 - beta1) * grad)
    Sl = (beta2 * s) + ((1 - beta2) * (grad ** 2))

    css = Ss / (1 - beta1 ** t)
    cSd = Sl / (1 - beta2 ** t)

    a = var - alpha * (css / ((cSd ** (0.5)) + epsilon))
    return a, Ss, Sl
