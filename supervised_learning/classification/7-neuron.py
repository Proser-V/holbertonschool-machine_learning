#!/usr/bin/env python3
"""
A module with a class that defines a single neuron performing binary
classification.
"""
import numpy as np
import matplotlib.pyplot as plt


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

        X: numpy.ndarray of shape (nx, m) containing the input data.
        """
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression.

        Y: a numpy.ndarray with shape (1, m) that contains the correct labels
        for the input data.
        A: a numpy.ndarray with shape (1, m) containing the activated output
        of the neuron for each example.
        """
        m = Y.shape[1]
        cost = - (1 / m) * np.sum(Y * np.log(A)
                                  + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neuron's predictions.

        X: a numpy.ndarray with shape (nx, m) that contains the input data.
        Y: a numpy.ndarray with shape (1, m) that contains the correct labels
        for the input data.
        """
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neuron.

        X: a numpy.ndarray with shape (nx, m) that contains the input data.
        Y: a numpy.ndarray with shape (1, m) that contains the correct labels
        for the input data.
        A: a numpy.ndarray with shape (1, m) containing the activated output
        of the neuron for each example.
        alpha: the learning rate
        """
        m = Y.shape[1]
        dZ = A - Y
        dW = (1 / m) * np.matmul(dZ, X.T)
        db = (1 / m) * np.sum(dZ)

        self.__W -= alpha * dW
        self.__b -= alpha * db

    def train(
            self, X, Y, iterations=5000,
            alpha=0.05, verbose=True, graph=True, step=100
            ):
        """
        Trains the neuron.

        X: a numpy.ndarray with shape (nx, m) that contains the input data.
        Y: a numpy.ndarray with shape (1, m) that contains the correct
        labels for the input data.
        iterations: the number of iterations to train over.
        alpha: the learning rate.
        verbose: a boolean that defines whether or not to print information
        about the training.
        graph: a boolean that defines whether or not to graph information
        about the training once the training has completed.
        step: number of iterations to go.
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        
        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        costs = []
        steps = []

        for i in range(iterations + 1):
            A = self.forward_prop(X)
            if i % step == 0 or i == iterations:
                cost = self.cost(Y, A)
                if verbose:
                    print(f"Cost after {i} iterations: {cost}")
                if graph:
                    steps.append(i)
                    costs.append(cost)
            if i < iterations:
                self.gradient_descent(X, Y, A, alpha)

        if graph:
            plt.plot(steps, costs, 'b-')
            plt.xlabel("iterations")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()

        return self.evaluate(X, Y)
