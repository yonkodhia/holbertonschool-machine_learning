#!/usr/bin/env python3
"""
specificity.py
"""

import numpy as np


def specificity(confusion):
    """calc the speci"""

    CL = np.diagonal(confusion)
    CA = np.sum(confusion, axis=1) - CL
    CM = np.sum(confusion, axis=0) - CL
    CK = np.sum(confusion) - (CL + CA + CM)
    specificity = CK / (CK + CM)

    return specificity
