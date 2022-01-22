#!/usr/bin/env python3

import numpy as np


def sensitivity(confusion):
    """function that Calculates the sens"""

    CL = np.diag(confusion)
    CA = np.sum(confusion, axis=1) - CL
    sensitivity = CL / (CL + CA)

    return sensitivity
