#!/usr/bin/env python3
"""
A module that contains a function that that performs element-wise
addition, subtraction, multiplication, and division.
"""


def np_elementwise(mat1, mat2):
    """
    The function that performs element-wise addition, subtraction,
    multiplication, and division.
    """
    sum = mat1 + mat2
    sub = mat1 - mat2
    mul = mat1 * mat2
    div = mat1 / mat2
    return sum, sub, mul, div
