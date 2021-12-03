#!/usr/bin/env python3
"""
multiplication of two matrix
"""


def mat_mul(mat1, mat2):
    if len(mat1[0]) != len(mat2):
        return None
    return [[sum(a * b for a, b in zip(x, y)) for y in zip(*mat2)]
            for x in mat1]
