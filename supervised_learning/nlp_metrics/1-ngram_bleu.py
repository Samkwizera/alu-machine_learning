#!/usr/bin/env python3
"""N-gram BLEU score"""
import numpy as np


def ngrams(sequence, n):
    """Builds the list of n-grams of a sequence"""
    return [tuple(sequence[i:i + n]) for i in range(len(sequence) - n + 1)]


def ngram_bleu(references, sentence, n):
    """Calculates the n-gram BLEU score for a sentence"""
    sent_grams = ngrams(sentence, n)
    counts = {}
    for gram in sent_grams:
        counts[gram] = counts.get(gram, 0) + 1

    max_ref = {}
    for ref in references:
        ref_counts = {}
        for gram in ngrams(ref, n):
            ref_counts[gram] = ref_counts.get(gram, 0) + 1
        for gram, count in ref_counts.items():
            max_ref[gram] = max(max_ref.get(gram, 0), count)

    clipped = 0
    for gram, count in counts.items():
        clipped += min(count, max_ref.get(gram, 0))

    precision = clipped / len(sent_grams)

    ref_len = min((abs(len(ref) - len(sentence)), len(ref))
                  for ref in references)[1]

    if len(sentence) > ref_len:
        bp = 1
    else:
        bp = np.exp(1 - ref_len / len(sentence))

    return bp * precision
