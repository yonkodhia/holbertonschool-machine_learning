#!/usr/bin/env python3
"""Advanced linear algebra module"""
import numpy as np


def definiteness(matrix):
    """
    Calculates the definetness of a matrix
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError('matrix must be a numpy.ndarray')
    if not np.all(matrix == matrix.T):
        return None
    try:
        eivals = np.linalg.eig(matrix)[0]
        if all(eivals >= 0):
            if any(eivals == 0):
                return 'Positive semi-definite'
            return 'Positive definite'
        if all(eivals <= 0):
            if any(eivals == 0):
                return 'Negative semi-definite'
            return 'Negative definite'
        return 'Indefinite'
    except Exception:
        return None
