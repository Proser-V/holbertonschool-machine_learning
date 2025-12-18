#!/usr/bin/env python3
"""
This module contains 3 classes linked to decision trees.
This task aim at find the depth of a decision tree.
"""
import numpy as np


class Node:
    """
    This class represent the node of a decision tree.
    """
    def __init__(self, feature=None, threshold=None,
                 left_child=None, right_child=None, is_root=False, depth=0):
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
        This function calculate the maximal depth of the tree by recursion.
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

    def count_nodes_below(self, only_leaves=False):
        """
        This method count the number of nodes.
        """
        left_count = self.left_child.count_nodes_below(only_leaves=only_leaves)
        right_count = self.right_child.count_nodes_below(
            only_leaves=only_leaves)

        if only_leaves:
            return left_count + right_count
        else:
            return 1 + left_count + right_count

    def left_child_add_prefix(self, text):
        """
        Function to add prefix to left child.
        """
        lines = text.split("\n")
        new_lines = ["    +--" + lines[0]]
        for x in lines[1:]:
            new_lines.append("    |  " + x)
        return "\n".join(new_lines)

    def right_child_add_prefix(self, text):
        """
        Function to add prefix to right child.
        """
        lines = text.split("\n")
        new_lines = ["    +--" + lines[0]]
        for x in lines[1:]:
            new_lines.append("       " + x)
        return "\n".join(new_lines)

    def __str__(self):
        """
        This method represent the object node.
        """
        if self.is_root:
            text = f"root [feature={self.feature}, threshold={self.threshold}]"
        else:
            text = (
                f"-> node [feature={self.feature}, "
                f"threshold={self.threshold}]"
            )

        parts = [text]

        if self.left_child is not None:
            parts.append(self.left_child_add_prefix(str(self.left_child)))

        if self.right_child is not None:
            parts.append(self.right_child_add_prefix(str(self.right_child)))

        return "\n".join(parts)

    def get_leaves_below(self):
        """
        This function gets the leaves below the current node.
        """
        leaves = []

        if self.left_child is not None:
            leaves.extend(self.left_child.get_leaves_below())

        if self.right_child is not None:
            leaves.extend(self.right_child.get_leaves_below())

        return leaves

    def update_bounds_below(self):
        """
        This function update the bounds of the tree.
        """
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -1*np.inf}

        for child in [self.left_child, self.right_child]:
            if child is None:
                continue

            child.lower = self.lower.copy()
            child.upper = self.upper.copy()

            f = self.feature
            t = self.threshold

            if child is self.left_child:
                child.lower[f] = max(child.lower.get(f, -np.inf), t)
            else:
                child.upper[f] = min(child.upper.get(f, np.inf), t)

        for child in [self.left_child, self.right_child]:
            if child is not None:
                child.update_bounds_below()

    def update_indicator(self):
        """
        This function take a numpy array and return a boolean size
        array.
        """
        def is_large_enough(x):
            """
            Check if all features are strictly upper than lower bounds.
            """
            if not hasattr(self, "lower") or len(self.lower) == 0:
                return np.ones(x.shape[0], dtype=bool)

            checks = np.array([
                np.greater(x[:, key], self.lower[key])
                for key in self.lower.keys()
            ])

            return np.all(checks, axis=0)

        def is_small_enough(x):
            """
            Check if all features are lower than upper bounds.
            """
            if not hasattr(self, "upper") or len(self.upper) == 0:
                return np.ones(x.shape[0], dtype=bool)

            checks = np.array([
                np.less_equal(x[:, key], self.upper[key])
                for key in self.upper.keys()
            ])

            return np.all(checks, axis=0)

        self.indicator = lambda x: np.all(np.array([is_large_enough(x),
                                                    is_small_enough(x)]),
                                          axis=0)


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
        Return the length of the leaf.
        """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """
        This function count the number of nodes below the leaf.
        """
        return 1

    def __str__(self):
        """
        The method to represent the leaf object.
        """
        return (f"-> leaf [value={self.value}]")

    def get_leaves_below(self):
        """
        This function check if there is a leaf below.
        """
        return [self]

    def update_bounds_below(self):
        """
        This function does nothing as it is in a leaf.
        """
        pass


class Decision_Tree():
    """
    This class represent the decision tree himself.
    """
    def __init__(self, max_depth=10, min_pop=1,
                 seed=0, split_criterion="random", root=None):
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
        Return the depth of the decision tree.
        """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """
        The method to count nodes.
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """
        This method represent the object in string.
        """
        return self.root.__str__() + "\n"

    def get_leaves(self):
        """
        This function get the leaves of a decision tree.
        """
        return self.root.get_leaves_below()

    def update_bounds(self):
        """
        This function update the bounds of the tree.
        """
        self.root.update_bounds_below()
