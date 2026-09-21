#!/usr/bin/env python3
"""Create bag-of-words embeddings for a collection of sentences."""

from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """Return the bag-of-words matrix and its ordered feature names."""
    vectorizer = CountVectorizer(vocabulary=vocab)
    embeddings = vectorizer.fit_transform(sentences).toarray()

    try:
        features = vectorizer.get_feature_names_out().tolist()
    except AttributeError:
        features = vectorizer.get_feature_names()

    return embeddings, features
