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

    def pred(self, x):
        """
        This function allow testing the predictions.
        """
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)


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

    def pred(self, x):
        """
        This function allow testing the predictions.
        """
        return self.value


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

    def pred(self, x):
        """
        This function allow testing the predictions.
        """
        return self.root.pred(x)

    def update_predict(self):
        """
        This function computes the prediction.
        """
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()
        self.predict = lambda A: np.sum(
            np.array([leaf.value * leaf.indicator(A) for leaf in leaves]),
            axis=0
        )

    def np_extrema(self, arr):
        """
        Compute the min and max values of numpy array.
        """
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """
        Randomly choose a splitting rule for a given node.
        """
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            feature_min, feature_max = self.np_extrema(
                self.explanatory[:, feature][node.sub_population])
            diff = feature_max-feature_min
        x = self.rng.uniform()
        threshold = (1 - x) * feature_min + x * feature_max
        return feature, threshold

    def fit(self, explanatory, target, verbose=0):
        """
        Train the decision tree on a given dataset.
        """
        if self.split_criterion == "random":
            self.split_criterion = self.random_split_criterion
        else:
            self.split_criterion = self.Gini_split_criterion
        self.explanatory = explanatory
        self.target = target
        self.root.sub_population = np.ones_like(self.target, dtype='bool')

        self.fit_node(self.root)

        self.update_predict()

        if verbose == 1:
            print(
                f"""  Training finished.
    - Depth                     : {self.depth()}
    - Number of nodes           : {self.count_nodes()}
    - Number of leaves          : {self.count_nodes(only_leaves=True)}
    - Accuracy on training data : {self.accuracy(self.explanatory,
                                   self.target)}"""
            )

    def fit_node(self, node):
        """
        Recursively build the decision tree from a given node.
        """
        node.feature, node.threshold = self.split_criterion(node)

        left_population = node.sub_population & (
            self.explanatory[:, node.feature] > node.threshold)
        right_population = node.sub_population & (
            self.explanatory[:, node.feature] <= node.threshold)

        # Is left node a leaf ?
        is_left_leaf = (
            (np.sum(left_population) < self.min_pop) or
            (node.depth + 1 == self.max_depth) or
            (np.unique(self.target[left_population]).size == 1)
        )

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        # Is right node a leaf ?
        is_right_leaf = (
            (np.sum(right_population) < self.min_pop) or
            (node.depth + 1 == self.max_depth) or
            (np.unique(self.target[right_population]).size == 1)
        )

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def get_leaf_child(self, node, sub_population):
        """
        Create a leaf node from a given sub-population.
        """
        values, counts = np.unique(
            self.target[sub_population], return_counts=True)
        value = values[np.argmax(counts)]
        leaf_child = Leaf(value)
        leaf_child.depth = node.depth+1
        leaf_child.sub_population = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """
        Create an internal child node from a given sub-population.
        """
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def accuracy(self, test_explanatory, test_target):
        """
        Compute the prediction accuracy of the decision tree.
        """
        return np.sum(np.equal(self.predict(
            test_explanatory), test_target))/test_target.size

    def Gini_split_criterion_one_feature(self, node, feature):
        """
        Compute the optimal Gini split for a given feature at a given node.
        """
        mask = node.sub_population
        x = (self.explanatory[mask, feature])
        y = self.target[mask]

        thresholds = self.possible_thresholds(node, feature)
        if thresholds.size == 0:
            return None, np.inf

        classes = np.unique(self.target)
        Y = (y[:, None] == classes[None, :])

        left_cond = (x[:, None] > thresholds[None, :])

        Left_F = left_cond[:, :, None] & Y[:, None, :]

        left_counts = Left_F.sum(axis=0).astype(float)
        left_n = left_counts.sum(axis=1)
        n = float(x.shape[0])

        total_counts = Y.sum(axis=0).astype(float)
        right_counts = total_counts[None, :] - left_counts
        right_n = n - left_n

        p_left = np.divide(left_counts, left_n[:, None],
                           out=np.zeros_like(left_counts, dtype=float),
                           where=(left_n[:, None] != 0))
        p_right = np.divide(right_counts, right_n[:, None],
                            out=np.zeros_like(right_counts, dtype=float),
                            where=(right_n[:, None] != 0))

        gini_left = 1.0 - np.sum(p_left ** 2, axis=1)
        gini_right = 1.0 - np.sum(p_right ** 2, axis=1)

        avg_gini = (left_n / n) * gini_left + (right_n / n) * gini_right

        j = np.argmin(avg_gini)
        return thresholds[j], avg_gini[j]

    def Gini_split_criterion(self, node):
        """
        Find the best Gini split among all features for a given node.
        """
        X = np.array([self.Gini_split_criterion_one_feature(node, i)
                      for i in range(self.explanatory.shape[1])])
        i = np.argmin(X[:, 1])
        return i, X[i, 0]

    def possible_thresholds(self, node, feature):
        """
        Helper function for possible thresholds.
        """
        values = np.unique((self.explanatory[:, feature])[node.sub_population])
        return (values[1:] + values[:-1]) / 2
