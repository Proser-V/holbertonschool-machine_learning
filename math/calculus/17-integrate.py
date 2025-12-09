#!/usr/bin/env python3
"""
Ths module contains a function that calculate the integral of
a polynomial.
"""


def poly_integral(poly, C=0):
    """
    The said function.
    """
    # validate C
    if not isinstance(C, int):
        return None

    # validate poly
    if not isinstance(poly, list) or len(poly) == 0:
        return None

    for coeff in poly:
        if not isinstance(coeff, (int, float)):
            return None

    # integration constant
    integral = [C]

    # integrate each term
    for i, coeff in enumerate(poly):
        new_coeff = coeff / (i + 1)

        if isinstance(new_coeff, float) and new_coeff.is_integer():
            new_coeff = int(new_coeff)

        integral.append(new_coeff)

    # Remove trailing zeros
    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()

    return integral
