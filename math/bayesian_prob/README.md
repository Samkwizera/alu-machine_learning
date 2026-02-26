# Bayesian Probability

This directory contains implementations of Bayesian probability functions applied to a clinical trial context — estimating the probability of a drug causing severe side effects.

## Files

### 0-likelihood.py
Calculates the likelihood of observing `x` patients with severe side effects out of `n` total, given hypothetical probabilities `P`.

**Function:** `likelihood(x, n, P)`
- **Args:**
  - `x` - number of patients that develop severe side effects (int)
  - `n` - total number of patients observed (int)
  - `P` - 1D numpy.ndarray of hypothetical probabilities of the side effect
- **Returns:** 1D numpy.ndarray of likelihoods for each probability in `P`

### 1-intersection.py
Calculates the intersection of obtaining the data with each hypothetical probability (likelihood × prior).

**Function:** `intersection(x, n, P, Pr)`
- **Args:**
  - `x` - number of patients that develop severe side effects (int)
  - `n` - total number of patients observed (int)
  - `P` - 1D numpy.ndarray of hypothetical probabilities
  - `Pr` - 1D numpy.ndarray of prior beliefs for each probability in `P`
- **Returns:** 1D numpy.ndarray of intersection values for each probability in `P`

### 2-marginal.py
Calculates the marginal probability of obtaining the observed data across all hypotheses.

**Function:** `marginal(x, n, P, Pr)`
- **Args:**
  - `x` - number of patients that develop severe side effects (int)
  - `n` - total number of patients observed (int)
  - `P` - 1D numpy.ndarray of hypothetical probabilities
  - `Pr` - 1D numpy.ndarray of prior beliefs for each probability in `P`
- **Returns:** float, the marginal probability of obtaining `x` and `n`

### 3-posterior.py
Calculates the posterior probability for each hypothetical probability given the observed data (Bayes' theorem).

**Function:** `posterior(x, n, P, Pr)`
- **Args:**
  - `x` - number of patients that develop severe side effects (int)
  - `n` - total number of patients observed (int)
  - `P` - 1D numpy.ndarray of hypothetical probabilities
  - `Pr` - 1D numpy.ndarray of prior beliefs for each probability in `P`
- **Returns:** 1D numpy.ndarray of posterior probabilities for each probability in `P`

## Requirements
- Python 3.x
- NumPy

## Usage

```python
import numpy as np
from likelihood import likelihood
from intersection import intersection
from marginal import marginal
from posterior import posterior

# Hypothetical probabilities and uniform priors
P = np.linspace(0, 1, 26)
Pr = np.ones(26) / 26

x = 26   # patients with side effects
n = 130  # total patients observed

print(likelihood(x, n, P))
print(intersection(x, n, P, Pr))
print(marginal(x, n, P, Pr))
print(posterior(x, n, P, Pr))
```