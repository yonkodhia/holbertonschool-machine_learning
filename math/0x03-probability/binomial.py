#!/usr/bin/env python3
""" Binomial"""


class Binomial:
    def __init__(self, data=None, n=1, p=0.5):
        """ init function """
        if data is not None:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) <= 2:
                raise ValueError("data must contain multiple values")
            mean = sum(data) / len(data)
            variance = 0
            for x in data:
                variance += (x - mean) ** 2
            variance = variance / len(data)
            p = 1 - (variance / mean)
            self.n = int(round(mean / p))
            self.p = float(mean / self.n)
        else:
            if n <= 0:
                raise ValueError("n must be a positive value")
            self.n = int(n)
            if (p <= 0) or (p >= 1):
                raise ValueError("p must be greater than 0 and less than 1")
            self.p = float(p)

    @staticmethod
    def factorial(n):
        """ function that calc the fact """
        factorial_num = 1
        for m in range(1, n + 1):
            factorial_num *= m
        return factorial_num

    def pmf(self, k):
        """ functio that cal the pmf """
        k = int(k)
        if (k < 0) or (k > self.n):
            return 0

        n_factorial = self.factorial(self.n)
        k_factorial = self.factorial(k)
        nk_factorial = self.factorial(self.n - k)
        binomial_coefficient = n_factorial / (nk_factorial * k_factorial)
        q = 1 - self.p
        return binomial_coefficient * ((self.p ** k) * (q ** (self.n - k)))

    def cdf(self, k):
        """ functio that cal the cdf"""
        if not 0 <= k <= self.n:
            return 0
        k = int(k)
        result = 0
        for i in range(k + 1):
            result += self.pmf(i)
        return result
