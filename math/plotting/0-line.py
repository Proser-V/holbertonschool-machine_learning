#!/usr/bin/env python3
"""
A module that contains a function that plots a line graph.
"""
import numpy as np
import matplotlib.pyplot as plt


def line():
    """
    The function that plots a line graph.
    """
    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))

    plt.plot(y, '-r')
    plt.xlim(0, 10)
    plt.show()
