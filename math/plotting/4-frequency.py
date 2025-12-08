#!/usr/bin/env python3
"""
A module that contains a function that plots a histogram.
"""
import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """
    The function that plots a histogram.
    """
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))

    bins = np.arange(0, 101, 10)
    plt.hist(student_grades, bins=bins, edgecolor="black")
    plt.title("Project A")

    plt.xlabel("Grades")
    plt.ylabel("Number of Students")

    plt.xlim(0, 100)
    plt.ylim(0, 30)
    plt.xticks(np.arange(0, 101, 10))
    plt.yticks(np.arange(0, 31, 5))

    plt.show()
