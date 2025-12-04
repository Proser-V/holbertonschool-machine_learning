#!/usr/bin/env python3
"""
A module that contains a function that concatenates two matrices
along a specific axis.
"""


def cat_matrices(mat1, mat2, axis=0):
    """Concatenates two matrices along a given axis."""

    if axis == 0:
        if not isinstance(mat1, list) or not isinstance(mat2, list):
            return None
        return mat1 + mat2

    if len(mat1) != len(mat2):
        return None

    result = []
    for sub1, sub2 in zip(mat1, mat2):
        merged = cat_matrices(sub1, sub2, axis=axis - 1)
        if merged is None:
            return None
        result.append(merged)

    return result
