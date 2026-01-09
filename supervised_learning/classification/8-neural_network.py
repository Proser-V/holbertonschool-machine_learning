#!/usr/bin/env python3
"""
A module with a class that defines a neural network with one hidden layer
performing binary classification.
"""
import numpy as np


class NeuralNetwork:
    """
    A class that defines a single neuron performing binary classification.
    """

    def __init__(self, nx, nodes):
        """
        Class constructor.

        nx: nx is the number of input features.
        nodes: the number of nodes found in the hidden layer.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        W1 = np.random.randn(1, nodes)
        b1 = 0
        A1 = 0
        W2 = np.random.randn(1, nx)
        b2 = 0
        A2 = 0
