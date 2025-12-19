#!/usr/bin/env python3
"""
This module contains 3 classes linked to decision trees.
This task aim at find the depth of a decision tree.
"""

import numpy as np

Node = __import__('8-build_decision_tree').Node
Leaf = __import__('8-build_decision_tree').Leaf


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
        Random splitting rule, safe against empty / constant sub-populations.

        Returns a (feature, threshold) pair. If no split is possible
        (all features constant or population too small), returns a
        harmless threshold to avoid infinite loops.
        """
        feature = 0
        threshold = 0.0
        idx = np.where(node.sub_population)[0]
        m = idx.size

        # If node is empty or has only one sample: no meaningful split
        if m <= 1:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            threshold = self.explanatory[idx[0], feature] if m == 1 else 0.0
            return feature, threshold

        Xn = self.explanatory[idx]          # (m, d)
        mins = Xn.min(axis=0)
        maxs = Xn.max(axis=0)
        diffs = maxs - mins

        valid = np.where(diffs > 0)[0]

        # No feature can split (all constant)
        if valid.size == 0:
            feature = self.rng.integers(0, Xn.shape[1])
            threshold = mins[feature]
            return feature, threshold

        # Choose a random splittable feature then a random threshold
        # in its range
        feature = valid[self.rng.integers(0, valid.size)]
        x = self.rng.uniform()
        threshold = (1 - x) * mins[feature] + x * maxs[feature]
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
        Function to fit nodes.
        """
        node.feature, node.threshold = self.random_split_criterion(node)

        left_population = node.sub_population & (
            self.explanatory[:, node.feature] > node.threshold
        )
        right_population = node.sub_population & (
            self.explanatory[:, node.feature] <= node.threshold
        )

        is_left_leaf = (
            (np.sum(left_population) <= self.min_pop) or
            (node.depth + 1 == self.max_depth)
        )

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        is_right_leaf = (
            (np.sum(right_population) <= self.min_pop) or
            (node.depth + 1 == self.max_depth)
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


class Random_Forest():
    """
    The random forest class.
    """
    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        self.numpy_predicts = []
        self.target = None
        self.numpy_preds = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.seed = seed

    def predict(self, explanatory):
        """
        Predict the target values for a given explanatory dataset using
        majority voting among all decision trees in the forest.
        """
        preds = np.array([pred(explanatory) for pred in self.numpy_preds])

        return np.apply_along_axis(
            lambda x: np.bincount(x.astype(int)).argmax(),
            axis=0,
            arr=preds
        )

    def fit(self, explanatory, target, n_trees=100, verbose=0):
        """
        The function that fit.
        """
        self.target = target
        self.explanatory = explanatory
        self.numpy_preds = []
        depths = []
        nodes = []
        leaves = []
        accuracies = []
        for i in range(n_trees):
            T = Decision_Tree(max_depth=self.max_depth,
                              min_pop=self.min_pop, seed=self.seed+i)
            T.fit(explanatory, target)
            self.numpy_preds.append(T.predict)
            depths.append(T.depth())
            nodes.append(T.count_nodes())
            leaves.append(T.count_nodes(only_leaves=True))
            accuracies.append(T.accuracy(T.explanatory, T.target))
        if verbose == 1:
            print(f"""  Training finished.
    - Mean depth                     : {np.array(depths).mean()}
    - Mean number of nodes           : {np.array(nodes).mean()}
    - Mean number of leaves          : {np.array(leaves).mean()}
    - Mean accuracy on training data : {np.array(accuracies).mean()}
    - Accuracy of the forest on td   : {self.accuracy(self.explanatory,
                                                      self.target)}""")

    def accuracy(self, test_explanatory, test_target):
        """
        Check the accuracy.
        """
        return np.sum(np.equal(self.predict(test_explanatory),
                               test_target))/test_target.size


class Isolation_Random_Tree():
    """
    Isolation Random Tree for unsupervised outlier detection.
    Each leaf's value corresponds to its depth; points falling into shallow
    leaves are more likely to be outliers.
    """
    def __init__(self, max_depth=10, seed=0, root=None):
        """
        Initialize the Isolation Random Tree.

        Parameters
        ----------
        max_depth : int
            Maximum depth allowed for the tree.
        seed : int
            Random seed for reproducibility.
        root : Node, optional
            Existing root node (default is None, in which case a new
            root is created).
        """
        self.rng = np.random.default_rng(seed)
        self.root = root if root else Node(is_root=True)
        self.explanatory = None
        self.max_depth = max_depth
        self.predict = None
        self.min_pop = 1

    def __str__(self):
        """
        Return a string representation of the tree structure.
        """
        return self.root.__str__() + "\n"

    def depth(self):
        """
        Return the maximum depth of the tree.

        Returns
        -------
        int
            Depth of the tree.
        """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """
        Count nodes in the tree.

        Parameters
        ----------
        only_leaves : bool
            If True, count only leaf nodes; otherwise count all nodes.

        Returns
        -------
        int
            Number of nodes in the tree.
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def update_bounds(self):
        """
        Update the bounds of all nodes in the tree.
        """
        self.root.update_bounds_below()

    def get_leaves(self):
        """
        Get all leaf nodes of the tree.

        Returns
        -------
        list of Leaf
            All leaves in the tree.
        """
        return self.root.get_leaves_below()

    def update_predict(self):
        """
        Define the predict function to return the depth of the leaf
        each sample falls into. Useful for outlier scoring.
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
        Compute the minimum and maximum of a numpy array.

        Parameters
        ----------
        arr : ndarray
            Array to analyze.

        Returns
        -------
        tuple
            (min_value, max_value)
        """
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """
        Randomly select a feature and threshold to split a node.
        Ensures split is valid even if node is small or constant.

        Parameters
        ----------
        node : Node
            Node to split.

        Returns
        -------
        tuple
            (feature_index, threshold_value)
        """
        feature = 0
        threshold = 0.0
        idx = np.where(node.sub_population)[0]
        m = idx.size

        if m <= 1:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            threshold = self.explanatory[idx[0], feature] if m == 1 else 0.0
            return feature, threshold

        Xn = self.explanatory[idx]
        mins = Xn.min(axis=0)
        maxs = Xn.max(axis=0)
        diffs = maxs - mins
        valid = np.where(diffs > 0)[0]

        if valid.size == 0:
            feature = self.rng.integers(0, Xn.shape[1])
            threshold = mins[feature]
            return feature, threshold

        feature = valid[self.rng.integers(0, valid.size)]
        x = self.rng.uniform()
        threshold = (1 - x) * mins[feature] + x * maxs[feature]
        return feature, threshold

    def get_leaf_child(self, node, sub_population):
        """
        Create a leaf child node.

        Parameters
        ----------
        node : Node
            Parent node.
        sub_population : ndarray of bool
            Mask indicating samples belonging to this leaf.

        Returns
        -------
        Leaf
            Leaf node with value equal to depth.
        """
        leaf_child = Leaf(value=node.depth + 1)
        leaf_child.depth = node.depth + 1
        leaf_child.sub_population = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """
        Create an internal child node.

        Parameters
        ----------
        node : Node
            Parent node.
        sub_population : ndarray of bool
            Mask indicating samples belonging to this child.

        Returns
        -------
        Node
            New internal node.
        """
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def fit_node(self, node):
        """
        Recursively fit a node and its children.

        Parameters
        ----------
        node : Node
            Node to fit.
        """
        node.feature, node.threshold = self.random_split_criterion(node)

        left_population = node.sub_population & (
            self.explanatory[:, node.feature] > node.threshold
        )
        right_population = node.sub_population & (
            self.explanatory[:, node.feature] <= node.threshold
        )

        is_left_leaf = ((np.sum(left_population) <= self.min_pop)
                        or (node.depth + 1 == self.max_depth))
        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        is_right_leaf = ((np.sum(right_population) <= self.min_pop)
                         or (node.depth + 1 == self.max_depth))
        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def fit(self, explanatory, verbose=0):
        """
        Fit the Isolation Random Tree to the dataset.

        Parameters
        ----------
        explanatory : ndarray
            Feature matrix of shape (n_samples, n_features).
        verbose : int, optional
            If 1, prints summary of the tree.
        """
        self.split_criterion = self.random_split_criterion
        self.explanatory = explanatory
        self.root.sub_population = np.ones(explanatory.shape[0], dtype=bool)
        self.fit_node(self.root)
        self.update_predict()

        if verbose:
            print(f"""  Training finished.
    - Depth                     : {self.depth()}
    - Number of nodes           : {self.count_nodes()}
    - Number of leaves          : {self.count_nodes(only_leaves=True)}""")
