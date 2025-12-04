#!/usr/bin/env python3
"""
This module contains a function that concatenate two matrices
along a specific axis.
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    The function to concatenate the two 2D matrices along
    a specific axis.
    """
    new_matrix = []
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        for row in mat1:
            new_matrix.append(row[:])
        for row in mat2:
            new_matrix.append(row[:])
        return new_matrix
    if axis == 1:
        if len(mat1) != len(mat2):
            return None
        for i in range(len(mat1)):
            new_row = mat1[i][:]
            new_row += mat2[i][:]
            new_matrix.append(new_row)
        return new_matrix

    return None
