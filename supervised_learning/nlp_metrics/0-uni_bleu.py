#!/usr/bin/env python3
"""Unigram BLEU score"""
import numpy as np


def uni_bleu(references, sentence):
    """Calculates the unigram BLEU score for a sentence"""
    counts = {}
    for word in sentence:
        counts[word] = counts.get(word, 0) + 1

    max_ref = {}
    for ref in references:
        ref_counts = {}
        for word in ref:
            ref_counts[word] = ref_counts.get(word, 0) + 1
        for word, count in ref_counts.items():
            max_ref[word] = max(max_ref.get(word, 0), count)

    clipped = 0
    for word, count in counts.items():
        clipped += min(count, max_ref.get(word, 0))

    precision = clipped / len(sentence)

    ref_len = min((abs(len(ref) - len(sentence)), len(ref))
                  for ref in references)[1]

    if len(sentence) > ref_len:
        bp = 1
    else:
        bp = np.exp(1 - ref_len / len(sentence))

    return bp * precision
