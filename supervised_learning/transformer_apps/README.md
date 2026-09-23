# Transformer Applications

This project applies the transformer built in the Attention project to a real
machine translation task: Portuguese to English, using the
`ted_hrlr_translate/pt_to_en` dataset from TensorFlow Datasets.

## Files

| File | Description |
|------|-------------|
| `0-dataset.py` | `Dataset` class that loads the data and builds sub-word tokenizers |
| `1-dataset.py` | Adds `encode` to turn a sentence pair into token sequences |
| `2-dataset.py` | Adds `tf_encode`, a graph-mode wrapper around `encode` |
| `3-dataset.py` | Adds the full input pipeline: filter, cache, shuffle, batch, prefetch |
| `4-create_masks.py` | Create the padding and look-ahead masks for training |
| `5-transformer.py` | The transformer network, self-contained |
| `5-train.py` | Train the transformer on the dataset |

## The pipeline

### Tokenization

Both languages get their own `SubwordTextEncoder`, built from the training
corpus with a target vocabulary of `2**15`. Sub-word tokenization keeps the
vocabulary small while still representing rare words, by splitting them into
pieces the tokenizer already knows.

Encoded sentences are wrapped in start and end tokens. Since the tokenizer
occupies indices `0` through `vocab_size - 1`, the two new tokens take the
next free indices:

```
start = vocab_size
end   = vocab_size + 1
```

### Graph-mode wrapping

`encode` calls `.numpy()` on its inputs, which only works eagerly. To use it
inside `tf.data.Dataset.map`, `tf_encode` wraps it in `tf.py_function` and then
calls `set_shape([None])` on each output — `py_function` returns tensors of
unknown shape, and the pipeline needs a rank to batch them.

### Batching

Training data is filtered to sentences within `max_len` tokens, cached,
shuffled, padded into batches, and prefetched. Validation data is only filtered
and batched — no shuffling, since evaluation order does not matter, and no
caching or prefetching.

Padding is what makes the masks necessary: every batch is rectangular, so
shorter sentences are filled with zeros that the model must learn to ignore.

## The masks

Three masks come out of `create_masks`, all using the convention that `1` marks
a position to hide:

- **encoder_mask** — `(batch, 1, 1, seq_len_in)`, hides padding in the input.
- **combined_mask** — `(batch, 1, seq_len_out, seq_len_out)`, the elementwise
  maximum of the target padding mask and a look-ahead mask. The look-ahead part
  stops each position from attending to later positions, which is what keeps
  the decoder from reading the answer it is meant to predict.
- **decoder_mask** — `(batch, 1, 1, seq_len_in)`, hides input padding in the
  decoder's second attention block, where it attends over the encoder output.

The extra middle dimensions exist so each mask broadcasts across the attention
heads.

## Training

The target sequence is offset by one: the decoder is fed `target[:, :-1]` and
is scored against `target[:, 1:]`, so at every position it predicts the next
token from the ones before it.

Loss is sparse categorical crossentropy with padded positions masked out and
the total divided by the number of real tokens, so padding does not dilute the
result. The optimizer is Adam with `beta_1=0.9`, `beta_2=0.98`, `epsilon=1e-9`
and the schedule from the original paper, which warms the learning rate up
linearly for 4000 steps and then decays it by the inverse square root of the
step:

```
lr = dm^-0.5 * min(step^-0.5, step * warmup_steps^-1.5)
```

## Usage

```python
#!/usr/bin/env python3
import tensorflow as tf
train_transformer = __import__('5-train').train_transformer

tf.compat.v1.set_random_seed(0)
transformer = train_transformer(4, 128, 8, 512, 32, 40, 2)
```

Two epochs is only enough to watch the loss fall. Real translation quality
needs 20 or more.

## Requirements

- Python 3.x
- NumPy
- TensorFlow
- TensorFlow Datasets
