#!/usr/bin/env python3
"""Calculate Shannon entropies ."""


import numpy as np


def HP(Di, beta):
    """Calculate hp function"""
    exponent = np.exp(-Di/beta)
    Pi = exponent / exponent.sum()
    Hi = -(Pi*np.log2(Pi)).sum()
    return Hi, Pi
