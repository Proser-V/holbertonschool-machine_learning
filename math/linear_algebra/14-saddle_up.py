#!/usr/bin/env python3
"""
A module that contains a function that performs matrix multiplication.
"""

import numpy as np


def np_matmul(mat1, mat2):
    """
    The function that performs matrix multiplication.
    """
    mat1 @ mat2
    return np.matmul(mat1, mat2)
