# Clustering

This directory contains implementations of common unsupervised learning
algorithms. It covers K-means clustering, Gaussian mixture models (GMMs),
expectation-maximization, model selection, and agglomerative clustering.

## Files

| File | Description |
|------|-------------|
| `0-initialize.py` | Initializes K-means centroids within the bounds of a data set |
| `1-kmeans.py` | Performs K-means clustering using Lloyd's algorithm |
| `2-variance.py` | Calculates total intra-cluster variance |
| `3-optimum.py` | Compares cluster counts using changes in variance |
| `4-initialize.py` | Initializes the priors, means, and covariances of a GMM |
| `5-pdf.py` | Evaluates a multivariate Gaussian probability density function |
| `6-expectation.py` | Performs the expectation step of the EM algorithm |
| `7-maximization.py` | Performs the maximization step of the EM algorithm |
| `8-EM.py` | Fits a Gaussian mixture model with expectation-maximization |
| `9-BIC.py` | Selects the best GMM cluster count using the Bayesian information criterion |
| `10-kmeans.py` | Performs K-means clustering with scikit-learn |
| `11-gmm.py` | Fits a Gaussian mixture model with scikit-learn |
| `12-agglomerative.py` | Performs Ward-linkage clustering and displays a dendrogram |

## Requirements

- Python 3
- NumPy
- SciPy
- Matplotlib
- scikit-learn

## Usage

The numbered modules can be imported with `__import__`:

```python
import numpy as np

kmeans = __import__('1-kmeans').kmeans

X = np.array([
    [1.0, 2.0],
    [1.5, 1.8],
    [8.0, 8.0],
    [9.0, 8.5],
])

centroids, classes = kmeans(X, 2)
print(centroids)
print(classes)
```

To fit a Gaussian mixture model with the custom EM implementation:

```python
expectation_maximization = __import__('8-EM').expectation_maximization

pi, means, covariances, probabilities, likelihood = \
    expectation_maximization(X, 2)
```

The functions return `None` values when their inputs do not satisfy the
required types or dimensions.
