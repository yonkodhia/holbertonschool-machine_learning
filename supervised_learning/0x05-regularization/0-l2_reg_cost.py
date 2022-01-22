#!/usr/bin/env python3
"""calculates the cost of a neural network with L2 regularization:"""

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """function l2_reg_cost"""

    if (L == 0):
        return 0

    lambtha
    result = 0

    for i in weights:
        if (i[0] == "W"):
            values = weights[i]
            result += np.linalg.norm(values)

    tot = cost + (lambtha / (2 * m)) * result
    return(tot)
