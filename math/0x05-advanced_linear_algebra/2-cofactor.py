#!/usr/bin/env python3
"""Advanced linear algebra module"""


def _det(matrix, total=0):
    """
    Calculates the determininant recursivly
    """
    if len(matrix) == 1:
        return matrix[0][0]

    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    indices = list(range(len(matrix)))

    for idx in indices:
        M = []
        for row in matrix[1:]:
            M.append(row.copy())

        height = len(M)
        for i in range(height):
            M[i] = M[i][0:idx] + M[i][idx+1:]

        sign = (-1) ** (idx % 2)
        total += sign * matrix[0][idx] * _det(M)

    return total


def cofactor(matrix):
    """Calculates the cofactor of matrix

    Args:
        matrix (list(list)): The given matrix

    Raises:
        TypeError: if the matrix is not a list of lists
        ValueError: if matrix is not a square matrix

    Returns:
        The minor matrix of the matrix

    """
    if not isinstance(matrix, list) or matrix == []:
        raise TypeError('matrix must be a list of lists')

    if any(not isinstance(row, list) for row in matrix):
        raise TypeError('matrix must be a list of lists')

    if any(len(row) != len(matrix) for row in matrix):
        raise ValueError('matrix must be a non-empty square matrix')

    if len(matrix) == 1:
        return [[1]]

    height = len(matrix)
    width = len(matrix[0])

    M = []
    for i in range(height):
        A = []
        for j in range(width):
            sub = []
            for row in (matrix[:i] + matrix[i + 1:]):
                sub.append(row[:j] + row[j + 1:])
            sign = (-1) ** ((i + j) % 2)
            A.append(_det(sub) * sign)
        M.append(A)
    return M
