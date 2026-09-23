#!/usr/bin/env python3
"""Loads and preps a dataset for machine translation."""

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset:
    """Loads and preps the ted_hrlr_translate/pt_to_en dataset."""

    def __init__(self):
        """Loads the train and validation splits and builds the tokenizers."""
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='train', as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='validation', as_supervised=True)
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

    def tokenize_dataset(self, data):
        """
        Creates sub-word tokenizers for the dataset

        data is a tf.data.Dataset whose examples are tuples (pt, en)

        Returns: tokenizer_pt, tokenizer_en
        """
        tokenizer_pt = tfds.deprecated.text.SubwordTextEncoder.\
            build_from_corpus((pt.numpy() for pt, en in data),
                              target_vocab_size=2**15)
        tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.\
            build_from_corpus((en.numpy() for pt, en in data),
                              target_vocab_size=2**15)
        return tokenizer_pt, tokenizer_en
