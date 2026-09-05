# Neural Style Transfer

This project implements the core components of neural style transfer using
TensorFlow and the pretrained VGG19 convolutional neural network. It combines
the content of one image with the visual style of another image.

## Files

| File | Description |
|------|-------------|
| `0-neural_style.py` | Validate and preprocess the style and content images |
| `1-neural_style.py` | Build a VGG19 feature extraction model |
| `2-neural_style.py` | Calculate normalized Gram matrices |
| `3-neural_style.py` | Extract target style and content features |
| `4-neural_style.py` | Calculate the style cost for a single layer |
| `5-neural_style.py` | Calculate the evenly weighted total style cost |

## NST Class

The `NST` class uses the following VGG19 layers:

- Style: `block1_conv1`, `block2_conv1`, `block3_conv1`, `block4_conv1`, and
  `block5_conv1`
- Content: `block5_conv2`

Input images must be NumPy arrays with shape `(height, width, 3)`. Images are
resized proportionally with bicubic interpolation so that their largest side
is 512 pixels, then normalized to the range `[0, 1]`.

## Requirements

- Python 3
- NumPy
- TensorFlow with Keras and VGG19 support
