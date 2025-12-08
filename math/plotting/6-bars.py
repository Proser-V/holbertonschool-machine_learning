#!/usr/bin/env python3
"""
A module that contains a function that plots a stacked bar graph.
"""
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """
    The function that plots a stacked bar graph.
    """
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))

    people = ['Farrah', 'Fred', 'Felicia']
    fruit_names = ['apples', 'bananas', 'oranges', 'peaches']
    colors = ['red', 'yellow', '#ff8000', '#ffe5b4']

    x = np.arange(len(people))
    width = 0.5

    bottom = np.zeros(len(people))

    for i in range(len(fruit_names)):
        plt.bar(x, fruit[i], width, bottom=bottom,
                color=colors[i], label=fruit_names[i])
        bottom += fruit[i]

    plt.ylabel('Quantity of Fruit')
    plt.title('Number of Fruit per Person')
    plt.xticks(x, people)
    plt.yticks(np.arange(0, 81, 10))
    plt.ylim(0, 80)

    plt.legend()
    plt.show()
