#!/usr/bin/env python3
"""Bayesian optimization acquisition module."""

import numpy as np
from scipy.stats import norm

GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """Performs Bayesian optimization on a noiseless 1D Gaussian process."""

    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1,
                 sigma_f=1, xsi=0.01, minimize=True):
        """Initialize Bayesian optimization."""
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """Calculate the next best sample location."""
        mu, sigma = self.gp.predict(self.X_s)

        if self.minimize:
            best = np.min(self.gp.Y)
            improvement = best - mu - self.xsi
        else:
            best = np.max(self.gp.Y)
            improvement = mu - best - self.xsi

        EI = np.zeros_like(mu)
        mask = sigma > 0
        Z = np.zeros_like(mu)
        Z[mask] = improvement[mask] / sigma[mask]
        cdf = norm.cdf(Z[mask])
        pdf = norm.pdf(Z[mask])
        EI[mask] = improvement[mask] * cdf + sigma[mask] * pdf

        X_next = self.X_s[np.argmax(EI)]
        return X_next, EI
