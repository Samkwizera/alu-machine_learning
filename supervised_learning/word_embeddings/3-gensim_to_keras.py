#!/usr/bin/env python3
"""Convert a Gensim Word2Vec model into a Keras embedding layer."""


def gensim_to_keras(model):
    """Return a trainable Keras embedding layer from ``model``."""
    return model.wv.get_keras_embedding(True)
