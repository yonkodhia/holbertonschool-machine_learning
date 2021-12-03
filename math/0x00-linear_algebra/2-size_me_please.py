#!/usr/bin/env python3
"""how to calculate a shape of matrix"""


def matrix_shape(matrix):
    x = [len(matrix)]
    while isinstance(matrix[0], list):
        x.append(len(matrix[0]))
        matrix = matrix[0]
    return x
