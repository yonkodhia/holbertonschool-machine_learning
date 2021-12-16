#!/usr/bin/env python3
"""Normal module"""


class Normal:
    """new class normal"""
    e = 2.7182818285
    pi = 3.1415926536

    def __init__(self, data=None, mean=0., stddev=1.):
        """function init"""
        if data is None:
            if stddev <= 0:
                raise ValueError('stddev must be a positive value')
            else:
                self.mean = float(mean)
                self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.mean = sum(data) / len(data)
            s = 0
            for item in data:
                s += (item - self.mean)**2
            self.stddev = (s / len(data))**.5

    def z_score(self, x):
        """function that calculate the x score and z value"""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """ calculate the x value"""
        return (z * self.stddev) + self.mean

    def pdf(self, x):
        """Calculates the PDF for a given x-value
        """
        z = self.z_score(x)
        return (1 / (self.stddev * (2 * Normal.pi)**.5))\
            * Normal.e**((-1/2) * z**2)

    def poww(self, x):
        """
        param x
        param self
        """
        o = (pow(x, 3))/3
        t = (pow(x, 5)) / 10
        th = (pow(x, 7))/42
        f = (pow(x, 9))/216
        return((2 / (Normal.pi ** 0.5)) * (x - o + t - th + f))

    def cdf(self, x):
        """Calculates the CDF for a given x-value
        x is the x-value
        Returns the CDF value for x
        """
        z = (self.z_score(x))/(pow(2, 1/2))
        res = 0.5*(1+self.poww(z))
        return(res)
