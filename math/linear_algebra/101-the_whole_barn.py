#!/usr/bin/env python3
"""
A module that contains a function that adds two matrices element-wise.
"""


def add_matrices(mat1, mat2):
    """Adds two matrices element-wise.
    """

    if type(mat1) is not type(mat2):
        return None
    if not isinstance(mat1, list):
        return mat1 + mat2
    if len(mat1) != len(mat2):
        return None

    result = []
    for a, b in zip(mat1, mat2):
        summed = add_matrices(a, b)
        if summed is None:
            return None
        result.append(summed)

    return result
