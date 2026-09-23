# NLP Metrics

This project implements the BLEU score from scratch, the standard metric for
evaluating machine-translated text against one or more reference translations.

## Files

| File | Description |
|------|-------------|
| `0-uni_bleu.py` | Calculate the unigram BLEU score for a sentence |
| `1-ngram_bleu.py` | Calculate the n-gram BLEU score for a sentence |
| `2-cumulative_bleu.py` | Calculate the cumulative n-gram BLEU score for a sentence |

## How BLEU works

BLEU combines two parts: **modified n-gram precision** and a **brevity
penalty**.

### Modified n-gram precision

Count how many n-grams of the candidate sentence appear in the references,
but clip each n-gram's count at the highest count it reaches in any single
reference. Clipping stops a candidate from inflating its score by repeating
a word that appears only once in the references.

```
precision = clipped matches / total candidate n-grams
```

### Brevity penalty

Precision alone rewards short output, so BLEU multiplies by a penalty when the
candidate is shorter than its reference. `r` is the length of the reference
closest to the candidate length `c` (ties go to the shorter reference):

```
BP = 1                 if c > r
BP = exp(1 - r / c)    otherwise
```

### Cumulative score

The cumulative score is the geometric mean of the precisions for n-grams of
size 1 through `n`, weighted evenly at `1/n` each, times the brevity penalty:

```
BLEU = BP * exp( (1/n) * sum(log p_i) )
```

## Usage

```python
#!/usr/bin/env python3

uni_bleu = __import__('0-uni_bleu').uni_bleu
ngram_bleu = __import__('1-ngram_bleu').ngram_bleu
cumulative_bleu = __import__('2-cumulative_bleu').cumulative_bleu

references = [["the", "cat", "is", "on", "the", "mat"],
              ["there", "is", "a", "cat", "on", "the", "mat"]]
sentence = ["there", "is", "a", "cat", "here"]

print(uni_bleu(references, sentence))            # 0.6549846024623855
print(ngram_bleu(references, sentence, 2))       # 0.6140480648084865
print(cumulative_bleu(references, sentence, 4))  # 0.5475182535069453
```

Each reference translation and the candidate sentence are passed as lists of
words, not as strings.

## Requirements

- Python 3.x
- NumPy
