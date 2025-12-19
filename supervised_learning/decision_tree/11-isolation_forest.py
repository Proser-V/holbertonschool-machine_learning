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
        Random splitting rule.
        """
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            feature_min, feature_max = self.np_extrema(
                self.explanatory[:, feature][node.sub_population]
            )
            diff = feature_max - feature_min

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

    def fit(self, explanatory, target, verbose=0):
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
        for i in range(self.n_trees):
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


class Isolation_Random_Tree:
    """
    A class that implements an isolation tree.
    """
    def __init__(self, max_depth=10, seed=0, root=None):
        """
        Initialize the Isolation_Random_Tree.
        """
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.max_depth = max_depth
        self.predict = None
        self.min_pop = 1

    def __str__(self):
        """
        String representation of the isolation tree.
        """
        return self.root.__str__() + "\n"

    def depth(self):
        """
        Get the depth of the isolation tree.
        """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """
        Count the nodes of the isolation tree.
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def update_bounds(self):
        """
        Update the bounds of the isolation tree.
        """
        self.root.update_bounds_below()

    def get_leaves(self):
        """
        Get the leaves of the isolation tree.
        """
        return self.root.get_leaves_below()

    def update_predict(self):
        """
        Update the predict of the isolation tree.
        """
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()

        self.predict = lambda A: np.array([self.root.pred(x) for x in A])

    def np_extrema(self, arr):
        """
        Get the minimum and maximum values of the array.
        """
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """
        Random split criterion for the isolation tree.
        """
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            feature_min, feature_max = self.np_extrema(
                self.explanatory[:, feature][node.sub_population])
            diff = feature_max - feature_min
        x = self.rng.uniform()
        threshold = (1 - x) * feature_min + x * feature_max
        return feature, threshold

    def get_leaf_child(self, node, sub_population):
        """
        Get the leaf child of the isolation tree.
        """
        leaf_child = Leaf(node.depth + 1)
        leaf_child.depth = node.depth + 1
        leaf_child.subpopulation = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """
        Get the node child of the isolation tree.
        """
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def fit_node(self, node):
        """
        Fit the node of the isolation tree.
        """
        node.feature, node.threshold = self.random_split_criterion(node)

        feature_values = np.greater(
            self.explanatory[:, node.feature],
            node.threshold)

        left_population = np.logical_and(
            node.sub_population,
            feature_values)
        right_population = np.logical_and(
            node.sub_population,
            np.logical_not(feature_values)
            )

        # Is left node a leaf ?
        is_left_leaf = np.any(np.array(
            [node.depth == self.max_depth - 1,
             np.sum(left_population) <= self.min_pop]))

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(
                node,
                left_population)
            self.fit_node(node.left_child)

        # Is right node a leaf ?
        is_right_leaf = np.any(np.array(
            [node.depth == self.max_depth - 1,
             np.sum(right_population) <= self.min_pop]))

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def fit(self, explanatory, verbose=0):
        """
        Fit the isolation tree to the data.
        """
        self.split_criterion = self.random_split_criterion
        self.explanatory = explanatory
        self.root.sub_population = np.ones_like(
            explanatory.shape[0],
            dtype='bool')

        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            print(f"""  Training finished.
    - Depth                     : {self.depth()}
    - Number of nodes           : {self.count_nodes()}
    - Number of leaves          : {self.count_nodes(only_leaves=True)}""")


class Isolation_Random_Forest():
    """
    Isolation Random Forest class.
    """
    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        """
        Initializes the isolation random forest.
        """
        self.numpy_predicts = []
        self.target = None
        self.numpy_preds = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.seed = seed

    def predict(self, explanatory):
        """
        Predicts the depth of the isolation random forest.
        """
        predictions = np.array([f(explanatory) for f in self.numpy_preds])
        return predictions.mean(axis=0)

    def fit(self, explanatory, n_trees=100, verbose=0):
        """
        Fits the isolation random forest to the data.
        """
        self.explanatory = explanatory
        self.numpy_preds = []
        depths = []
        nodes = []
        leaves = []
        for i in range(n_trees):
            T = Isolation_Random_Tree(
                max_depth=self.max_depth,
                seed=self.seed+i)
            T.fit(explanatory)
            self.numpy_preds.append(T.predict)
            depths.append(T.depth())
            nodes.append(T.root.count_nodes_below())
            leaves.append(T.root.count_nodes_below(only_leaves=True))
        if verbose == 1:
            print(f"""  Training finished.
    - Mean depth                     : {np.array(depths).mean()}
    - Mean number of nodes           : {np.array(nodes).mean()}
    - Mean number of leaves          : {np.array(leaves).mean()}""")

    def suspects(self, explanatory, n_suspects):
        """
        Returns the n_suspects rows in explanatory that have
        the smallest mean depth
        """
        depths = self.predict(explanatory)
        suspect_indices = np.argsort(depths)[:n_suspects]
        suspects = explanatory[suspect_indices]
        suspect_depths = depths[suspect_indices]
        return suspects, suspect_depths
