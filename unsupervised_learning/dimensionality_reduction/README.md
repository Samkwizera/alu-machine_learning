# Dimensionality Reduction

This directory contains exercises that use principal component analysis
(PCA) to reduce the number of features in a data set while preserving its
most important directions.

## Files

| File | Description |
|------|-------------|
| `0-pca.py` | Calculates a PCA weight matrix using a variance threshold |
| `1-pca.py` | Centers and transforms a data set to a fixed dimensionality |

## Requirements

- Python 3
- NumPy

## Usage

The numbered modules can be imported with `__import__`.

### Selecting dimensions by variance

The first implementation expects a data set whose feature means are already
zero. It returns the weight matrix needed to retain the requested fraction of
the data's variance.

```python
import numpy as np

pca = __import__('0-pca').pca

X = np.array([
    [2.0, 0.0, 1.0],
    [0.0, 2.0, -1.0],
    [-2.0, 0.0, 1.0],
    [0.0, -2.0, -1.0],
])

W = pca(X, var=0.95)
transformed = np.matmul(X, W)
```

### Transforming to a fixed dimensionality

The second implementation centers the input data before returning its PCA
transformation with exactly `ndim` components.

```python
pca = __import__('1-pca').pca

transformed = pca(X, ndim=2)
print(transformed.shape)
```
