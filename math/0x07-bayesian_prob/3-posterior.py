#!/usr/bin/env python3
"""posterior probabilities"""


import numpy as np
import scipy.special as special


def posterior(x, n, P, Pr):
    """posterior function"""
    margin = marginal(x, n, P, Pr)
    return likelihood(x, n, P) * Pr / margin


def marginal(x, n, P, Pr):
    """marginal function"""
    likely = likelihood(x, n, P)
    if np.where(Pr > 1, 1, 0).any() or np.where(Pr < 0, 1, 0).any():
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if type(Pr) is not np.ndarray or P.shape != Pr.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if not np.isclose(Pr.sum(), 1):
        raise ValueError("Pr must sum to 1")
    return (likely * Pr).sum()


def intersection(x, n, P, Pr):
    """intersection function"""
    if type(n) is not int or n <= 0:
        raise ValueError("n must be a positive integer")
    if type(x) is not int or x <= 0:
        raise ValueError("x must be an integer that is "
                         "greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if type(P) is not np.ndarray or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if type(Pr) is not np.ndarray or P.shape != Pr.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if np.where(P > 1, 1, 0).any() or np.where(P < 0, 1, 0).any():
        raise ValueError("All values in P must be in the range [0, 1]")
    if np.where(Pr > 1, 1, 0).any() or np.where(Pr < 0, 1, 0).any():
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(Pr.sum(), 1):
        raise ValueError("Pr must sum to 1")
    return likelihood(x, n, P) * Pr


def likelihood(x, n, P):
    """likelihood function"""
    if type(n) is not int or n <= 0:
        raise ValueError("n must be a positive integer")
    if type(x) is not int or x < 0:
        raise ValueError("x must be an integer that "
                         "is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if type(P) is not np.ndarray or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if np.where(P > 1, 1, 0).any() or np.where(P < 0, 1, 0).any():
        raise ValueError("All values in P must be in the range [0, 1]")
    return special.binom(n, x) * pow(P, x) * pow(1 - P, n - x)
