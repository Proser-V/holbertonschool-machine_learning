#!/usr/bin/env python3
"""
This module contains a function that that calculates a summ.
"""


def summation_i_squared(n):
    """
    The function that calculates a summ.
    """
    if type(n) not in [int, float] or n != int(n) or n < 1:
        return None
    n = int(n)
    return n * (n + 1) * (2 * n + 1) // 6
