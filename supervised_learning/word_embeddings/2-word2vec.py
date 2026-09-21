#!/usr/bin/env python3
"""Create and train a Word2Vec model."""

from gensim.models import Word2Vec


def word2vec_model(sentences, size=100, min_count=5, window=5,
                   negative=5, cbow=True, iterations=5, seed=0,
                   workers=1):
    """Create and train a Word2Vec model from tokenized sentences.

    Args:
        sentences: List of tokenized sentences used for training.
        size: Dimensionality of the word vectors.
        min_count: Minimum frequency required for a word.
        window: Maximum distance between current and predicted words.
        negative: Number of negative samples.
        cbow: If True, use CBOW; otherwise, use skip-gram.
        iterations: Number of passes over the corpus.
        seed: Random number generator seed.
        workers: Number of worker threads.

    Returns:
        The trained ``gensim.models.Word2Vec`` model.
    """
    common = {
        "min_count": min_count,
        "window": window,
        "negative": negative,
        "sg": 0 if cbow else 1,
        "seed": seed,
        "workers": workers,
    }

    try:
        return Word2Vec(sentences, vector_size=size, epochs=iterations,
                        **common)
    except TypeError:
        return Word2Vec(sentences, size=size, iter=iterations, **common)
