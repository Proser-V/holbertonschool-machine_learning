#!/usr/bin/env python3
"""
This module contains a function that add two matrices
element-wise.
"""


def add_matrices2D(mat1, mat2):
    """
    The function to add two matrices element-wise.
    """
    if not mat1 and not mat2:
        return []

    if (not mat1 and mat2) or (mat1 and not mat2):
        return None

    if len(mat1) != len(mat2):
        return None

    if len(mat1[0]) != len(mat2[0]):
        return None

    new_matrix = [
        [mat1[i][j] + mat2[i][j] for j in range(len(mat1[0]))]
        for i in range(len(mat1))
    ]

    return new_matrix
