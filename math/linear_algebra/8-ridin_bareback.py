#!/usr/bin/env python3
"""
A module that contains a function that performs matrix multiplication.
"""


def mat_mul(mat1, mat2):
    """
    The function to perform the matrix multiplication.
    """
    if len(mat1[0]) != len(mat2):
        return None
    new_matrix = [[0]*len(mat2[0]) for _ in range(len(mat1))]
    for i in range(len(mat1)):
        for j in range(len(mat2[0])):
            for k in range(len(mat2)):
                new_matrix[i][j] += mat1[i][k] * mat2[k][j]
    return new_matrix
