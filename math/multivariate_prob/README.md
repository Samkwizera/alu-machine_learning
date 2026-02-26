# Multivariate Probability

This directory contains implementations of multivariate probability and statistics functions.

## Files

### 0-mean_cov.py
Calculates the mean and covariance matrix of a data set.

**Function:** `mean_cov(X)`
- **Args:** `X` - numpy.ndarray of shape (n, d) containing the data set
- **Returns:** 
  - `mean` - numpy.ndarray of shape (1, d) containing the mean
  - `cov` - numpy.ndarray of shape (d, d) containing the covariance matrix

### 1-correlation.py
Calculates a correlation matrix from a covariance matrix.

**Function:** `correlation(C)`
- **Args:** `C` - numpy.ndarray of shape (d, d) containing a covariance matrix
- **Returns:** numpy.ndarray of shape (d, d) containing the correlation matrix

### multinormal.py
Implements a class representing a Multivariate Normal distribution.

**Class:** `MultiNormal`
- **Constructor:** `__init__(self, data)`
  - **Args:** `data` - numpy.ndarray of shape (d, n) containing the data set
  - Sets instance attributes: `mean` and `cov`

- **Method:** `pdf(self, x)`
  - Calculates the PDF at a data point
  - **Args:** `x` - numpy.ndarray of shape (d, 1) containing the data point
  - **Returns:** float value of the PDF at x

## Requirements
- Python 3.x
- NumPy

## Usage

```python
import numpy as np
from multinormal import MultiNormal

# Create sample data
data = np.random.randn(3, 100)

# Initialize MultiNormal distribution
mn = MultiNormal(data)

# Calculate PDF at a point
x = np.array([[0], [0], [0]])
pdf_value = mn.pdf(x)
```
