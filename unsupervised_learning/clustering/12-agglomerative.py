#!/usr/bin/env python3
"""Agglomerative clustering with a dendrogram."""

import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """Cluster ``X`` with Ward linkage at cophenetic distance ``dist``."""
    linkage = scipy.cluster.hierarchy.linkage(X, method='ward')
    scipy.cluster.hierarchy.dendrogram(linkage, color_threshold=dist)
    plt.show()
    return scipy.cluster.hierarchy.fcluster(
        linkage, t=dist, criterion='distance')
