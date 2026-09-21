#!/usr/bin/env python3
"""Create and train a FastText model."""

from gensim.models import FastText


def fasttext_model(sentences, size=100, min_count=5, negative=5,
                   window=5, cbow=True, iterations=5, seed=0,
                   workers=1):
    """Create and train a FastText model from tokenized sentences.

    Args:
        sentences: List of tokenized sentences used for training.
        size: Dimensionality of the word vectors.
        min_count: Minimum frequency required for a word.
        negative: Number of negative samples.
        window: Maximum distance between current and predicted words.
        cbow: If True, use CBOW; otherwise, use skip-gram.
        iterations: Number of passes over the corpus.
        seed: Random number generator seed.
        workers: Number of worker threads.

    Returns:
        The trained ``gensim.models.FastText`` model.
    """
    common = {
        "min_count": min_count,
        "negative": negative,
        "window": window,
        "sg": 0 if cbow else 1,
        "seed": seed,
        "workers": workers,
    }

    try:
        return FastText(sentences, vector_size=size, epochs=iterations,
                        **common)
    except TypeError:
        return FastText(sentences, size=size, iter=iterations, **common)
