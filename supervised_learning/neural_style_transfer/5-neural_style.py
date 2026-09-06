#!/usr/bin/env python3
"""Utilities for neural style transfer."""

import numpy as np
import tensorflow as tf

tf.enable_eager_execution()


class NST:
    """Perform neural style transfer tasks."""

    style_layers = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
        'block5_conv1'
    ]
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """Initialize a neural style transfer instance."""
        if not (isinstance(style_image, np.ndarray) and
                style_image.ndim == 3 and style_image.shape[2] == 3):
            raise TypeError(
                'style_image must be a numpy.ndarray with shape (h, w, 3)'
            )
        if not (isinstance(content_image, np.ndarray) and
                content_image.ndim == 3 and content_image.shape[2] == 3):
            raise TypeError(
                'content_image must be a numpy.ndarray with shape (h, w, 3)'
            )
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError('alpha must be a non-negative number')
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError('beta must be a non-negative number')

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.load_model()
        self.generate_features()

    @staticmethod
    def scale_image(image):
        """Scale an image so its largest side is 512 pixels."""
        if not (isinstance(image, np.ndarray) and
                image.ndim == 3 and image.shape[2] == 3):
            raise TypeError(
                'image must be a numpy.ndarray with shape (h, w, 3)'
            )

        height, width = image.shape[:2]
        scale = 512 / max(height, width)
        new_shape = (int(height * scale), int(width * scale))
        image = tf.convert_to_tensor(image, dtype=tf.float32)
        image = tf.expand_dims(image, axis=0)
        image = tf.image.resize_bicubic(image, new_shape)

        return tf.clip_by_value(image / 255.0, 0.0, 1.0)

    def load_model(self):
        """Create the VGG19 feature extraction model."""
        vgg = tf.keras.applications.VGG19(
            include_top=False,
            weights='imagenet'
        )
        layer_names = self.style_layers + [self.content_layer]
        outputs = []
        output = vgg.input

        for layer in vgg.layers[1:]:
            if isinstance(layer, tf.keras.layers.MaxPooling2D):
                layer = tf.keras.layers.AveragePooling2D(
                    pool_size=layer.pool_size,
                    strides=layer.strides,
                    padding=layer.padding,
                    name=layer.name
                )
            output = layer(output)
            if layer.name in layer_names:
                outputs.append(output)
            if layer.name == self.content_layer:
                break

        self.model = tf.keras.Model(vgg.input, outputs)
        self.model.trainable = False

    @staticmethod
    def gram_matrix(input_layer):
        """Calculate the normalized Gram matrix of a layer output."""
        if (not isinstance(input_layer, (tf.Tensor, tf.Variable)) or
                len(input_layer.shape) != 4):
            raise TypeError('input_layer must be a tensor of rank 4')

        gram = tf.linalg.einsum(
            'bijc,bijd->bcd', input_layer, input_layer
        )
        dimensions = tf.shape(input_layer)
        locations = tf.cast(dimensions[1] * dimensions[2],
                            input_layer.dtype)
        return gram / locations

    def generate_features(self):
        """Extract the style and content features from the images."""
        style_input = tf.keras.applications.vgg19.preprocess_input(
            self.style_image * 255.0
        )
        content_input = tf.keras.applications.vgg19.preprocess_input(
            self.content_image * 255.0
        )

        style_outputs = self.model(style_input)
        content_outputs = self.model(content_input)

        self.gram_style_features = [
            self.gram_matrix(output) for output in style_outputs[:-1]
        ]
        self.content_feature = content_outputs[-1]

    def layer_style_cost(self, style_output, gram_target):
        """Calculate the style cost for one model layer."""
        if (not isinstance(style_output, (tf.Tensor, tf.Variable)) or
                len(style_output.shape) != 4):
            raise TypeError('style_output must be a tensor of rank 4')

        channels = style_output.shape[-1]
        target_shape = [1, channels, channels]
        if not isinstance(gram_target, (tf.Tensor, tf.Variable)) or \
                gram_target.shape.as_list() != target_shape:
            raise TypeError(
                'gram_target must be a tensor of shape [1, {}, {}]'.format(
                    channels, channels
                )
            )

        gram_style = self.gram_matrix(style_output)
        return tf.reduce_mean(tf.square(gram_style - gram_target))

    def style_cost(self, style_outputs):
        """Calculate the evenly weighted style cost for all layers."""
        layer_count = len(self.style_layers)
        if not isinstance(style_outputs, list) or \
                len(style_outputs) != layer_count:
            raise TypeError(
                'style_outputs must be a list with a length of {}'.format(
                    layer_count
                )
            )

        costs = [
            self.layer_style_cost(output, target)
            for output, target in zip(
                style_outputs, self.gram_style_features
            )
        ]
        return tf.add_n(costs) / layer_count
