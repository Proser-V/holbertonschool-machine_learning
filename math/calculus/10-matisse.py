#!/usr/bin/env python3
"""
This module contains a function that calculates the derivative of a polynomia.
"""


def poly_derivative(poly):
    """
    The function that calculates the derivative of a polynomia.
    """
    if (not isinstance(poly, list)
            or len(poly) == 0
            or not all( isinstance(num, (int, float)) for num in poly)):
        return None

    if len(poly) == 1:
        return [0]

    new_poly = []
    for i in range(1, len(poly)):
        new_poly.append(i * poly[i])

    if all(num == 0 for num in new_poly):
        return [0]
