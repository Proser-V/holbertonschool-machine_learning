#!/usr/bin/env python3
"""
This module contains a function that sums corresponding elements
of two arrays
"""


def add_arrays(arr1, arr2):
    """
    The function to sum matching elements of arrays
    """
    if len(arr1) != len(arr2):
        return None
    new_array = []
    for i in range(len(arr1)):
        new_array.append(arr1[i] + arr2[i])

    return new_array
