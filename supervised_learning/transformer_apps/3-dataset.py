#!/usr/bin/env python3
"""Sets up the data pipeline for machine translation."""

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset:
    """Loads, preps, and batches the ted_hrlr_translate/pt_to_en dataset."""

    def __init__(self, batch_size, max_len):
        """
        batch_size is the batch size for training/validation
        max_len is the maximum number of tokens allowed per example sentence
        """
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='train', as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='validation', as_supervised=True)
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

        def max_length(pt, en):
            """Keeps only the examples within max_len tokens."""
            return tf.logical_and(tf.size(pt) <= max_len,
                                  tf.size(en) <= max_len)

        self.data_train = self.data_train.filter(max_length)
        self.data_train = self.data_train.cache()
        self.data_train = self.data_train.shuffle(20000)
        self.data_train = self.data_train.padded_batch(batch_size)
        self.data_train = self.data_train.prefetch(
            tf.data.experimental.AUTOTUNE)

        self.data_valid = self.data_valid.filter(max_length)
        self.data_valid = self.data_valid.padded_batch(batch_size)

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

    def encode(self, pt, en):
        """
        Encodes a translation into tokens

        pt is the tf.Tensor containing the Portuguese sentence
        en is the tf.Tensor containing the corresponding English sentence

        Returns: pt_tokens, en_tokens
        """
        pt_vocab = self.tokenizer_pt.vocab_size
        en_vocab = self.tokenizer_en.vocab_size

        pt_tokens = [pt_vocab] + self.tokenizer_pt.encode(
            pt.numpy()) + [pt_vocab + 1]
        en_tokens = [en_vocab] + self.tokenizer_en.encode(
            en.numpy()) + [en_vocab + 1]

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """
        Acts as a tensorflow wrapper for the encode instance method

        Returns: pt_tokens, en_tokens as tensors with their shapes set
        """
        pt_tokens, en_tokens = tf.py_function(func=self.encode,
                                              inp=[pt, en],
                                              Tout=[tf.int64, tf.int64])
        pt_tokens.set_shape([None])
        en_tokens.set_shape([None])
        return pt_tokens, en_tokens
