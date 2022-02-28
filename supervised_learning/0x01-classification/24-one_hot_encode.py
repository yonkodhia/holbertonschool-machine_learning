#!/usr/bin/env python3
"""One-Hot encoding module"""
import numpy as np


def one_hot_encode(Y, classes):
    """Converts a numerical label vector into one-hot matrix

    Args:
        Y (numpy.ndarray): Is the numeric class labels with shape (m,) where
        m is the number of examples
        classes (int): Is the maximum number of classes founds in Y.

    Returns:
        numpy.ndarray|None: Returns a one-hot encoding of Y with shape
        (classes, m), or None in failure.

    """
    if type(Y) != np.ndarray:
        return None
    try:
        return np.eye(classes)[Y].T
    except Exception:
        return None
