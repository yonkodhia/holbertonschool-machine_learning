#!/usr/bin/env python3
"""add matrices"""


def add_matrices(mat1, mat2):
    """Recursively construct a new sum of two matrices"""
    matrices = []
    x = len(mat1)
    y = len(mat2)
    if x != y:
        return None
    for i in mat1:
        for j in mat2:
            new = add_matrices(i, j)
    if new is None:
        return None
    matrices.append(new)
    return matrices
