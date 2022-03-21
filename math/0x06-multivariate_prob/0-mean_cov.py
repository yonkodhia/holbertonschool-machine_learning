#!/usr/bin/env python3
"""Multivariate probability module"""
import numpy as np


def mean_cov(X):
    """Calculates the mean and covariance of data set

    Args:
        X (np.ndarray): of shape (n, d) containing the data set where n is
            the number of data point, and d is the number of dimension.

    Raises:
        TypeError: If X is not a 2D numpy.ndarray
        ValueError: If n is less than 2

    Returns:
        np.ndarray: of shape (1, d) represent the mean
        np.ndarray: of shape (d, d) represent the covariance matrix

    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        raise TypeError('X must be a 2D numpy.ndarray')

    n, d = X.shape
    if n < 2:
        raise ValueError('X must contain multiple data points')

    mean = np.expand_dims(np.mean(X, axis=0), axis=0)
    cov = np.matmul((X - mean).T, (X - mean)) / (n - 1)
    return mean, cov
