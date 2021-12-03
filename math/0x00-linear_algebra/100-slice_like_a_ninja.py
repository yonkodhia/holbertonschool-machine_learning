#!/usr/bin/env python3
"""slices a matrix along specific axes"""


def np_slice(matrix, axes={}):
    """function numpy slices"""
    array = [slice(None)] * (max(axes) + 1)
    for i, j in axes.items():
        array[i] = slice(*j)
    return matrix[tuple(array)]
