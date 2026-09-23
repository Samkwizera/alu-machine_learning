# Attention

This project builds attention mechanisms from the ground up, starting with
additive attention over an RNN encoder-decoder and ending with a complete
transformer network.

## Files

| File | Description |
|------|-------------|
| `0-rnn_encoder.py` | `RNNEncoder` class that encodes a source sequence with a GRU |
| `1-self_attention.py` | `SelfAttention` class implementing additive (Bahdanau) attention |
| `2-rnn_decoder.py` | `RNNDecoder` class that decodes one word at a time using attention |
| `4-positional_encoding.py` | Calculate the positional encoding for a transformer |
| `5-sdp_attention.py` | Calculate the scaled dot product attention |
| `6-multihead_attention.py` | `MultiHeadAttention` class splitting attention across heads |
| `7-transformer_encoder_block.py` | `EncoderBlock` class for a single transformer encoder block |
| `8-transformer_decoder_block.py` | `DecoderBlock` class for a single transformer decoder block |
| `9-transformer_encoder.py` | `Encoder` class stacking N encoder blocks |
| `10-transformer_decoder.py` | `Decoder` class stacking N decoder blocks |
| `11-transformer.py` | `Transformer` class joining the encoder, decoder, and output layer |

The numbering follows the project spec, which skips `3-`.

## Concepts

### Additive attention (files 0-2)

The encoder runs a GRU over the source sentence and keeps every hidden state
rather than just the last one. At each decoding step the alignment model
scores the previous decoder state against all encoder states:

```
score   = V(tanh(W(s_prev) + U(hidden_states)))
weights = softmax(score, axis=1)
context = sum(weights * hidden_states, axis=1)
```

The softmax runs over the sequence axis, so the weights form a distribution
across source positions. The context vector is concatenated with the embedded
previous target word and fed to the decoder GRU.

### Scaled dot product attention (file 5)

The transformer replaces the alignment network with a dot product, scaled by
`sqrt(dk)` to keep the softmax out of its saturated region:

```
output = softmax(Q @ K.T / sqrt(dk)) @ V
```

A mask, when supplied, is multiplied by `-1e9` and added before the softmax so
masked positions receive effectively zero weight.

### Multi-head attention (file 6)

Rather than attending once with `dm` dimensions, the input is projected and
split into `h` heads of `depth = dm / h` dimensions each. Every head attends
independently, letting the model track several kinds of relationship at once.
The heads are then concatenated and passed through a final linear layer.

### Positional encoding (file 4)

Attention is permutation invariant, so position must be injected explicitly.
Each position gets a vector of alternating sines and cosines across a
geometric range of frequencies:

```
PE[pos, 2i]     = sin(pos / 10000^(2i / dm))
PE[pos, 2i + 1] = cos(pos / 10000^(2i / dm))
```

### Blocks and full network (files 7-11)

Each encoder block is multi-head self attention followed by a feed-forward
network, both wrapped in residual connections and layer normalization. Decoder
blocks add a second attention layer that attends over the encoder output. The
`Transformer` stacks N of each and projects the decoder output to the target
vocabulary. Embeddings are scaled by `sqrt(dm)` before the positional encoding
is added.

## Requirements

- Python 3.x
- NumPy
- TensorFlow
