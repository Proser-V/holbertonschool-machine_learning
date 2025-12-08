#!/usr/bin/env python3
"""
A module that contains a function that plots all 5 previous graphs.
"""
import numpy as np
import matplotlib.pyplot as plt


def all_in_one():
    """
    The function that plots all 5 previous graphs
    """
    y0 = np.arange(0, 11) ** 3

    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
    y1 += 180

    x2 = np.arange(0, 28651, 5730)
    r2 = np.log(0.5)
    t2 = 5730
    y2 = np.exp((r2 / t2) * x2)

    x3 = np.arange(0, 21000, 1000)
    r3 = np.log(0.5)
    t31 = 5730
    t32 = 1600
    y31 = np.exp((r3 / t31) * x3)
    y32 = np.exp((r3 / t32) * x3)

    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)

    # Global figure
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle('All in One')

    # Graph 1
    plt1 = plt.subplot(3, 2, 1)
    plt1.plot(y0, "-r")
    plt1.set_xlim(0, 10)
    plt1.tick_params(labelsize="x-small")

    # Graph 2
    plt2 = plt.subplot(3, 2, 2)
    plt2.scatter(x1, y1, marker='o', color="m")
    plt2.set_title("Men's Height vs Weight", fontsize="x-small")
    plt2.set_xlabel("Height (in)", fontsize="x-small")
    plt2.set_ylabel("Weight (lbs)", fontsize="x-small")
    plt2.tick_params(labelsize="x-small")

    # Graph 3
    plt3 = plt.subplot(3, 2, 3)
    plt3.plot(x2, y2)
    plt3.set_title("Exponential Decay of C-14", fontsize="x-small")
    plt3.set_xlabel("Time (years)", fontsize="x-small")
    plt3.set_ylabel("Fraction Remaining", fontsize="x-small")
    plt3.set_xlim(0, 28650)
    plt3.set_yscale("log")
    plt3.tick_params(labelsize="x-small")

    # Graph 4
    plt4 = plt.subplot(3, 2, 4)
    plt4.set_title("Exponential Decay of Radioactive Elements",
                   fontsize="x-small")
    plt4.plot(x3, y31, color="red", linestyle="--", label="C-14")
    plt4.plot(x3, y32, color="green", linestyle="-", label="Ra-226")
    plt4.set_xlabel("Time (years)", fontsize="x-small")
    plt4.set_ylabel("Fraction Remaining", fontsize="x-small")
    plt4.set_xlim(0, 20000)
    plt4.set_ylim(0, 1)
    plt4.legend(fontsize="x-small")
    plt4.tick_params(labelsize="x-small")

    # Graph 5
    plt5 = plt.subplot(3, 2, (5, 6))
    bins = np.arange(0, 101, 10)
    plt5.hist(student_grades, bins=bins, edgecolor="black")
    plt5.set_title("Project A", fontsize="x-small")
    plt5.set_xlabel("Grades", fontsize="x-small")
    plt5.set_ylabel("Number of Students", fontsize="x-small")
    plt5.set_xlim(0, 100)
    plt5.set_ylim(0, 30)
    plt5.set_xticks(np.arange(0, 101, 10))
    plt5.set_yticks(np.arange(0, 31, 5))
    plt5.tick_params(labelsize="x-small")

    plt.tight_layout()
    plt.show()
