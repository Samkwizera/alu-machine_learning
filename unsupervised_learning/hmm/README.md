# Hidden Markov Models

This directory contains NumPy implementations of Markov chain and Hidden
Markov Model (HMM) algorithms.

## Requirements

- Python 3
- NumPy
- All transition and emission matrices must contain valid probability
  distributions.

## Files

| File | Description |
| --- | --- |
| `0-markov_chain.py` | Calculates a Markov chain's state distribution after a specified number of transitions. |
| `1-regular.py` | Determines the steady-state distribution of a regular Markov chain. |
| `2-absorbing.py` | Determines whether a Markov chain is absorbing. |
| `3-forward.py` | Uses the forward algorithm to calculate the likelihood of an observation sequence. |
| `4-viterbi.py` | Finds the most likely sequence of hidden states with the Viterbi algorithm. |
| `5-backward.py` | Uses the backward algorithm to calculate the likelihood of an observation sequence. |
| `6-baum_welch.py` | Estimates transition and emission probabilities with the Baum-Welch algorithm. |

## Markov chain example

Because the filenames begin with numbers, the modules can be loaded with
`importlib`:

```python
import importlib
import numpy as np

markov_chain = importlib.import_module("0-markov_chain").markov_chain

P = np.array([
    [0.25, 0.20, 0.25, 0.30],
    [0.20, 0.30, 0.20, 0.30],
    [0.25, 0.25, 0.40, 0.10],
    [0.30, 0.30, 0.10, 0.30],
])
s = np.array([[1, 0, 0, 0]])

print(markov_chain(P, s, 300))
```

Expected output:

```text
[[0.2494929  0.26335362 0.23394185 0.25321163]]
```

## HMM data shapes

- `Observation`: `(T,)`, containing observation indexes.
- `Emission`: `(N, M)`, where rows are hidden states and columns are possible
  observations.
- `Transition`: `(N, N)`, containing hidden-state transition probabilities.
- `Initial`: `(N, 1)`, containing initial hidden-state probabilities.

Invalid inputs return `None` values, or `False` for `absorbing`.
