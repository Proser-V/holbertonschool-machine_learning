#!/usr/bin/env python3
"""
A module to define the depth of a tree.
"""
import numpy as np


class Node:
    """
    This class represent a node of a decision tree.
    """
    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """
        This methods define the max depth of the decision tree.
        """
        if self.left_child is None and self.right_child is None:
            return self.depth

        if self.left_child is not None:
            left_max = self.left_child.max_depth_below()
        else:
            left_max = self.depth

        if self.right_child is not None:
            right_max = self.right_child.max_depth_below()
        else:
            right_max = self.depth

        return max(left_max, right_max)


class Leaf(Node):
    """
    This class represent the leaf of a decision tree.
    """
    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """
        Docstring for max_depth_below.
        """
        return self.depth


class Decision_Tree():
    """
    This class represent the decision tree himself.
    """
    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """
        This method returns the depth.
        """
        return self.root.max_depth_below()
