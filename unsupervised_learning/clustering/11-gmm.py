#!/usr/bin/env python3
"""Gaussian mixture modeling via scikit-learn."""

import sklearn.mixture


def gmm(X, k):
    """Fit a full-covariance Gaussian mixture model with ``k`` components."""
    model = sklearn.mixture.GaussianMixture(n_components=k)
    model.fit(X)
    clss = model.predict(X)
    return (model.weights_, model.means_, model.covariances_, clss,
            model.bic(X))
