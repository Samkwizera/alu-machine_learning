#!/usr/bin/env python3
"""Create TF-IDF embeddings for a collection of sentences."""

from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """Return the TF-IDF matrix and its ordered feature names."""
    vectorizer = TfidfVectorizer(vocabulary=vocab)
    embeddings = vectorizer.fit_transform(sentences).toarray()

    try:
        features = vectorizer.get_feature_names_out().tolist()
    except AttributeError:
        features = vectorizer.get_feature_names()

    return embeddings, features
