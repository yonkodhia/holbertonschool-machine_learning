#!/usr/bin/env python3
"""Piosson module"""


class Poisson:
    """function poiss
    """
    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """def Initializer"""
        if data is None:
            if lambtha <= 0:
                raise ValueError('lambtha must be a positive value')
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.lambtha = sum(data) / len(data)

    def pmf(self, k):
        """function calc
        """
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0
        else:
            return (self.lambtha**k * Poisson.e**((-1) * self.lambtha))\
                / Poisson.factorial(k)

    def cdf(self, k):
        """Calculates the value
        """
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0
        s = 0
        for a in range(0, k+1):
            s += self.lambtha**a / Poisson.factorial(a)
        return Poisson.e**((-1) * self.lambtha) * s

    @staticmethod
    def factorial(k):
        """Calculates the fact
        """
        f = 1
        for a in range(1, k + 1):
            f *= a
        return f
