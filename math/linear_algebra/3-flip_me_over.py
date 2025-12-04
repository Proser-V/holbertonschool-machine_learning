#!/usr/bin/env python3
"""
A module that contains the function to transpose a 2D matrix.
"""


def matrix_transpose(matrix):
    """
    A function function to transpose a 2D matrix.
    """
    new_matrix = [[matrix[i][j] for i in range(len(matrix))]
                  for j in range(len(matrix[0]))]
    return new_matrix
