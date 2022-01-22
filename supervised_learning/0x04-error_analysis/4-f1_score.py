#!/usr/bin/env python3
"""
4-f1_score.py
"""

import numpy as np

sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """calc the f1 score"""
    su = sensitivity(confusion)
    pre = precision(confusion)
    F_1_score = 2 * ((pre * su) / (pre + su))

    return F_1_score
