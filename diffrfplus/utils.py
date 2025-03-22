"""
TODO
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Inspired from an implementation of the Diff-RF algorithm provided at
# https://github.com/pfmarteau/DiFF-RF


import numpy as np



def calculate_nbins(size: float) -> float:
    """
    TODO
    """
    return int(size / 8) + 2

def generate_feature_distribution(data):
    """
    TODO
    """
    nbins = calculate_nbins(len(data))

    feature_distribution = []
    for i in range(np.shape(data)[1]):
        feature_distribution.append(weight_feature(data[:, i], nbins))
    feature_distribution = np.array(feature_distribution)

    return feature_distribution/(feature_distribution.sum() + 1e-5)

def split(column):
    """
    TODO
    """
    xmin = column.min()
    xmax = column.max()
    return np.random.uniform(xmin, xmax)

def similarity_score(instances, node, alpha):
    """
    TODO
    """
    d = np.shape(instances)[1]
    if len(instances) > 0:
        d = np.shape(instances)[1]
        similarity = (instances - node.avg) / node.std
        similarity = 2**(-alpha * (np.sum((similarity * similarity)/d, axis=1)))
    else:
        similarity = 0
    return similarity


def empirical_entropy(hist):
    """
    TODO
    """
    h = np.asarray(hist, dtype=np.float64)
    if h.sum() <= 0 or (h < 0).any():
        return 0
    h = h/h.sum()
    return -(h * np.ma.log2(h)).sum()


def weight_feature(s, nbins):
    """
    TODO
    """
    wmin = .02
    mins = s.min()
    maxs = s.max()

    if not np.isfinite(mins) or not np.isfinite(maxs) or np.abs(mins- maxs) < 1e-10:
        return 1e-4
    if mins == maxs:
        return 1e-4

    hist, _ = np.histogram(s, bins=nbins)
    ent = empirical_entropy(hist) / np.log2(nbins)

    if np.isfinite(ent):
        return max(1-ent, wmin)

    return wmin
