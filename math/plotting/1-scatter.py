#!/usr/bin/env python3
"""
A module that contains a function that plots a scatter plot.
"""
import numpy as np
import matplotlib.pyplot as plt


def scatter():
    """
    The function that plots a scatter plot.
    """
    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x, y = np.random.multivariate_normal(mean, cov, 2000).T
    y += 180
    plt.figure(figsize=(6.4, 4.8))

    plt.scatter(x, y, marker='o', color="m")
    plt.title("Men's Height vs Weight")
    plt.xlabel("Height (in)")
    plt.ylabel("Weight (lbs)")

    plt.show()
