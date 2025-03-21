"""
TODO
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Inspired from an implementation of the Diff-RF algorithm provided at
# https://github.com/pfmarteau/DiFF-RF

import random
import numpy as np

from .. import utils
from . import clustering

class Node:
    """
    TODO
    """
    def __init__(self, data, height_limit, feature_distribution=None, sample_size=None, height=0, is_leaf=False):
        """
        TODO
        """
        self.size = len(data)
        self.algorithm = 'hdbscan'
        self.height_limit = height_limit
        self.height       = height

        self.feature_distribution = feature_distribution

        self.is_leaf = is_leaf
        self.n_nodes = 1
        self.sample_size = sample_size
        self.freq = self.size / sample_size if sample_size else None

        if is_leaf:
            self._initialize_leaf(data)
        else:
            self._initialize_internal(data)

    def _initialize_internal(self, data):
        """
        TODO
        """
        if len(data) > 32:
            self.feature_distribution = utils.generate_feature_distribution(data)

        cols = np.arange(np.shape(data)[1], dtype='int')
        self.split_feature = random.choices(cols, weights=self.feature_distribution)[0]
        split_column = data[:, self.split_feature]

        self.split_value = utils.split(split_column)

        mask = split_column <= self.split_value
        data_left = data[mask, :]

        next_height = self.height + 1
        limit_reached = (self.height_limit <= self.height or
                         data_left.shape[0] <= 16 or
                         np.all(data_left.max(0) == data_left.min(0)))

        self.left = Node(data_left if data_left.shape[0] else data,
                         self.height_limit, self.feature_distribution,
                         self.sample_size, next_height,
                         is_leaf=limit_reached)

        data_right = data[~mask, :]
        limit_reached = (self.height_limit <= self.height or
                         data_right.shape[0] <= 16 or
                         np.all(data_right.max(0) == data_right.min(0)))

        self.right = Node(data_right if data_right.shape[0] else data,
                         self.height_limit, self.feature_distribution,
                         self.sample_size, next_height,
                         is_leaf=limit_reached)

        self.n_nodes = 1 + self.left.n_nodes + self.right.n_nodes

    def _initialize_leaf(self, data):
        """
        TODO
        """
        self.cluster_model, labels = clustering.build_cluster(data, algorithm=self.algorithm)

        if np.any(labels == -1):
            labels[labels == -1] = np.max(labels) + 1

        # Averge per cluster.
        clusters = [np.where(labels == label)[0].tolist() for label in np.unique(labels)]

        self.n_clusters = len(clusters)
        self.avg = np.zeros((self.n_clusters, np.shape(data)[1]))
        self.std = np.ones((self.n_clusters, np.shape(data)[1]))

        for label, cluster in enumerate(clusters):
            self.avg[label] = np.mean(data[cluster, :], axis=0)

        if len(data) > 10:
            for label, cluster in enumerate(clusters):
                self.std[label] = np.std(data[cluster, :], axis=0)
            self.std[self.std == 0] = 1e-2

    def similarity_score(self, instances, alpha):
        """
        TODO
        """
        d = np.shape(instances)[1]
        labels = self.cluster_model.predict(instances)

        if len(instances) > 0:
            avg = self.avg[labels]
            std = self.std[labels]
            similarity = (instances - avg) / std
            similarity = 2**(-alpha * (np.sum((similarity * similarity)/d, axis=1)))
        else:
            similarity = 0
        return similarity
