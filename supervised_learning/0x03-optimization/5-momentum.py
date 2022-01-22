#!/usr/bin/env python3
"""
Momentum
"""

import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """ function to update variables"""

    g = grad
    n = var

    li = beta1 * v + (1 - beta1) * g
    n = n - (alpha * li)

    return n, li
