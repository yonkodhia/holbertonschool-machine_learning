#!/usr/bin/env python3
"""
1-normalize
"""

import numpy as np


def normalize(X, m, s):
    """function that normalize matrix"""
    rs = (X - m) / s

    return rs
