#!/usr/bin/env python3
"""
traspose matrix
"""


def matrix_transpose(matrix):
    m = [[matrix[j][i] for j in range(len(matrix))]
         for i in range(len(matrix[0]))]
    while isinstance(matrix, list):
        return m
