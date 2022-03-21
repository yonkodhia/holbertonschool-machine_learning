#!/usr/bin/env python3
"""Multivariate probability module"""
import numpy as np


def correlation(C):
    """Calculates the correlation matrix

    Args:
        C (np.ndarray): of shape (d, d) containing the covariance matrix

    Raises:
        TypeError: If the C is not a numpy.ndarray
        ValueError: If the C doesn't have the shape (d, d)

    Returns:
        numpy.ndarray: of shape (d, d) containing the correlation matrix

    """
    if not isinstance(C, np.ndarray):
        raise TypeError('C must be a numpy.ndarray')
    if len(C.shape) != 2 or C.shape[0] != C.shape[1]:
        raise ValueError('C must be a 2D square matrix')

    d, d = C.shape
    D = np.sqrt(np.diag(C))
    D = np.eye(d) * D
    D_inv = np.linalg.inv(D)
    return np.matmul(np.matmul(D_inv, C), D_inv)
