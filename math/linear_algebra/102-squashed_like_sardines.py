#!/usr/bin/env python3
"""
A module that contains a function that concatenates two matrices
along a specific axis.
"""


def get_shape(mat):
    """Return shape tuple of nested lists, or None if ragged."""

    if not isinstance(mat, list):
        return ()
    if len(mat) == 0:
        return (0,)
    first = get_shape(mat[0])
    if first is None:
        return None
    for el in mat:
        if get_shape(el) != first:
            return None
    return (len(mat),) + first


def cat_matrices(mat1, mat2, axis=0):
    """Concatenate two nested-list matrices along axis or return None."""

    if not isinstance(mat1, list) or not isinstance(mat2, list):
        return None

    s1 = get_shape(mat1)
    s2 = get_shape(mat2)
    if s1 is None or s2 is None:
        return None

    if axis < 0 or axis >= len(s1):
        return None

    if axis == 0:
        if s1[1:] != s2[1:]:
            return None
        return list(mat1) + list(mat2)

    if len(mat1) != len(mat2):
        return None

    result = []
    for a, b in zip(mat1, mat2):
        merged = cat_matrices(a, b, axis - 1)
        if merged is None:
            return None
        result.append(merged)
    return result
