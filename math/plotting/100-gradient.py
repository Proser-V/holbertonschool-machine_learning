#!/usr/bin/env python3
"""
A module that contains a function to create a scatter plot of sampled
elevations on a mountain.
"""
import numpy as np
import matplotlib.pyplot as plt


def gradient():
    """
    The function to create a scatter plot of sampled elevations on a mountain.
    """
    np.random.seed(5)

    x = np.random.randn(2000) * 10
    y = np.random.randn(2000) * 10
    z = np.random.rand(2000) + 40 - np.sqrt(np.square(x) + np.square(y))
    plt.figure(figsize=(6.4, 4.8))

    sc = plt.scatter(x, y, c=z, cmap="viridis")

    plt.title("Mountain Elevation")
    plt.xlabel("x coordinate (m)")
    plt.ylabel("y coordinate (m)")

    cbar = plt.colorbar(sc)
    cbar.set_label("elevation (m)")

    plt.show()
