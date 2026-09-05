#!/usr/bin/env python3
"""Utilities for neural style transfer."""

import numpy as np
import tensorflow as tf


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

        if not tf.executing_eagerly():
            tf.compat.v1.enable_eager_execution()

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

        image = tf.convert_to_tensor(image, dtype=tf.float32)
        shape = tf.cast(tf.shape(image)[:2], tf.float32)
        scale = 512.0 / tf.reduce_max(shape)
        new_shape = tf.cast(shape * scale, tf.int32)
        image = tf.image.resize(image, new_shape, method='bicubic')
        image = tf.expand_dims(image, axis=0)

        return tf.clip_by_value(image / 255.0, 0.0, 1.0)

    def load_model(self):
        """Create the VGG19 feature extraction model."""
        vgg = tf.keras.applications.VGG19(
            include_top=False,
            weights='imagenet'
        )

        def replace_pooling(layer):
            """Clone a VGG layer, replacing max pooling with average."""
            if isinstance(layer, tf.keras.layers.MaxPooling2D):
                return tf.keras.layers.AveragePooling2D.from_config(
                    layer.get_config()
                )
            return layer.__class__.from_config(layer.get_config())

        base_model = tf.keras.models.clone_model(
            vgg,
            clone_function=replace_pooling
        )
        base_model.set_weights(vgg.get_weights())
        base_model.trainable = False

        layer_names = self.style_layers + [self.content_layer]
        outputs = [base_model.get_layer(name).output
                   for name in layer_names]
        self.model = tf.keras.Model(base_model.input, outputs)

    @staticmethod
    def gram_matrix(input_layer):
        """Calculate the normalized Gram matrix of a layer output."""
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)) or \
                input_layer.shape.rank != 4:
            raise TypeError('input_layer must be a tensor of rank 4')

        gram = tf.linalg.einsum('bijc,bijd->bcd',
                               input_layer, input_layer)
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
        if not isinstance(style_output, (tf.Tensor, tf.Variable)) or \
                style_output.shape.rank != 4:
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
