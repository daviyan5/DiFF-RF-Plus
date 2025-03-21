"""
TODO
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Inspired from an implementation of the Diff-RF algorithm provided at
# https://github.com/pfmarteau/DiFF-RF
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from hdbscan import HDBSCAN, prediction

INDEX = 0

class EmptyModel:
    """
    A placeholder class to represent an empty model.
    This is used when there is not enough data to build a model.
    """
    def fit(self, data):
        pass

    def predict(self, data):
        return np.zeros(data.shape[0], dtype=int)


class HDBSCANWrapper:
    def __init__(self, *args, **kwargs):
        self.model = HDBSCAN(*args, **kwargs)

    def fit(self, data):
        self.model.fit(data)
        return self

    def fit_predict(self, data):
        self.model.fit(data)
        return self.model.labels_

    def predict(self, data):
        print(self.model.labels_)
        labels = prediction.approximate_predict(self.model, data)[0]
        return labels

def build_cluster(data, algorithm='kmeans'):
    """
    Cluster the given data using the specified algorithm.

    Parameters
    ----------
    data : np.ndarray
        A (N x M) array of features.
    algorithm : str, optional
        The clustering algorithm to use. Options are:
         - 'kmeans': uses KMeans 
         - 'dbscan': uses DBSCAN
         - 'hdbscan': uses HDBSCAN
         
    Returns
    -------
    labels : np.ndarray
        An array of cluster labels assigned to each point in data.
        Outliers/noise (if any) are typically labeled as -1.
    """
    algo = algorithm.lower()
    if data.shape[0] <= 1:
        return EmptyModel(), np.zeros(data.shape[0], dtype=int)
    if algo == 'kmeans':
        model = KMeans(n_clusters=min(2, len(data)))
        model.fit(data)
        labels = model.labels_
    elif algo == 'dbscan':
        model = DBSCAN(eps=0.5, min_samples=min(3, len(data)))
        model.fit(data)
        labels = model.labels_
    elif algo == 'hdbscan':
        model = HDBSCANWrapper(min_cluster_size=min(3, len(data)), allow_single_cluster=True, prediction_data=True)
        labels = model.fit_predict(data)
    else:
        raise ValueError(f"Unsupported clustering algorithm: {algorithm}")
    return model, labels
