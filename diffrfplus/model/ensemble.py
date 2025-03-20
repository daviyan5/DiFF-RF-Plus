"""
TODO
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Inspired from an implementation of the Diff-RF algorithm provided at
# https://github.com/pfmarteau/DiFF-RF

from multiprocessing import Pool
from functools import partial

import numpy as np

from .. import utils
from .tree import Node

class TreeEnsemble:
    """
    TODO
    """
    def __init__(self, sample_size: int, n_trees: int = 10):
        self.sample_size = sample_size
        self.n_trees = n_trees
        self.alpha = 1.0

        self.data = None
        self.trees = None
        self.feature_distribution = []
        self.test_size = 1

        self.distance_scores = np.zeros((1, self.n_trees))
        self.frequency_scores = np.zeros((1, self.n_trees))
        self.collective_scores = np.zeros((1, self.n_trees))

    @staticmethod
    def calculate_height_limit(sample_size: float) -> float:
        """
        TODO
        """
        return 1.0 * np.ceil(np.log2(sample_size))

    def create_trees(self, data, feature_distribution: np.ndarray, sample_size: int, height_limit: float) -> Node:
        """
        TODO
        """
        rows = np.random.choice(len(data), sample_size, replace=False)
        return Node(data[rows, :], height_limit, feature_distribution, sample_size=sample_size)

    def fit(self, data: np.ndarray, n_jobs: int = 1):
        """
        TODO
        """
        self.data = data

        self.sample_size = min(self.sample_size, len(data))

        height_limit = self.calculate_height_limit(self.sample_size)
        self.feature_distribution = utils.generate_feature_distribution(data)

        create_tree_partial = partial(self.create_trees,
                                      feature_distribution= self.feature_distribution,
                                      sample_size=self.sample_size,
                                      height_limit=height_limit)

        with Pool(n_jobs) as p:
            self.trees = p.map(create_tree_partial, [data for _ in range(self.n_trees)])
        return self


    def walk(self, data: np.ndarray) -> np.ndarray:
        """
        TODO
        """

        self.distance_scores.resize((len(data), self.n_trees))
        self.frequency_scores.resize((len(data), self.n_trees))
        self.collective_scores.resize((len(data), self.n_trees))

        # We can of course parallelize this loop
        for tree_idx, tree in enumerate(self.trees):
            # This is highly inefficient,
            # because it's using a array to map the valid indices at each split
            cur_idx = np.ones(len(data)).astype(bool)
            self.walk_tree(tree, tree_idx, cur_idx,
                           data, self.feature_distribution, alpha=self.alpha)

    def walk_tree(self, node, tree_idx, cur_idx, data, feature_distribution, alpha):
        """
        TODO
        """
        if node.is_leaf:
            instances = data[cur_idx]
            f = ((node.size+1)/self.sample_size) / ((1+len(instances)) / self.test_size)
            if alpha == 0:
                self.distance_scores[cur_idx, tree_idx] = 0
                self.frequency_scores[cur_idx, tree_idx] = -f
                self.collective_scores[cur_idx, tree_idx] = -f
            else:
                z = utils.similarity_score(instances, node, alpha)
                self.distance_scores[cur_idx, tree_idx] = z
                self.frequency_scores[cur_idx, tree_idx] = -f
                self.collective_scores[cur_idx, tree_idx] = z*f

        else:

            left_idx = (data[:, node.splitAtt] <= node.splitValue) * cur_idx
            self.walk_tree(node.left, tree_idx, left_idx, data, feature_distribution, alpha)

            right_idx = (data[:, node.splitAtt] > node.splitValue) * cur_idx
            self.walk_tree(node.right, tree_idx, right_idx, data, feature_distribution, alpha)


    def anomaly_score(self, data: np.ndarray, alpha = 1) -> np.ndarray:
        """
        TODO
        """
        self.test_size = len(data)
        self.alpha     = alpha

        # Evaluate the scores for each of the observations.
        self.walk(data)

        # Compute the scores from the path lengths (self.L)
        scores = {
            'distance':   self.distance_scores.mean(1),
            'frequency':  self.frequency_scores.mean(1),
            'collective': self.collective_scores.mean(1)
        }
        return scores

    def predict_from_anomaly_scores(self, scores: np.ndarray, threshold: float) -> np.ndarray:
        """
        TODO
        """
        out = scores >= threshold
        return out * 1

    def predict(self, data: np.ndarray, threshold: float, score_type: str = 'distance') -> np.ndarray:
        """
        TODO
        """
        if score_type not in ['distance', 'frequency', 'collective']:
            raise RuntimeError('Invalid score type. Please choose from: \'distance\', \'frequency\', \'collective\'')
        scores = self.anomaly_score(data)
        return self.predict_from_anomaly_scores(scores[score_type], threshold)
