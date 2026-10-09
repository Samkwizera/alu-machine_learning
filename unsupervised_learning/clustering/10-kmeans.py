#!/usr/bin/env python3
"""K-means clustering via scikit-learn."""

import sklearn.cluster


def kmeans(X, k):
    """Cluster ``X`` using scikit-learn's KMeans implementation."""
    model = sklearn.cluster.KMeans(n_clusters=k)
    clss = model.fit_predict(X)
    return model.cluster_centers_, clss
