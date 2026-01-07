#!/usr/bin/env python3
"""
A module with a class that defines a single neuron performing binary
classification.
"""
import numpy as np


class Neuron:
    """
    A class that defines a single neuron performing binary classification.
    """

    def __init__(self, nx):
        """
        Class constructor.

        nx: nx is the number of input features to the neuron.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """
        The weights vector for the neuron.
        """
        return self.__W

    @property
    def b(self):
        """
        The bias for the neuron.
        """
        return self.__b

    @property
    def A(self):
        """
        The activated output of the neuron (prediction).
        """
        return self.__A

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron.

        X: numpy.ndarray of shape (nx, m) containing the input data
        """
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A
