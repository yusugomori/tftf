import numpy as np
import tensorflow as tf
from .Layer import Layer
from .initializers import *


class Conv2D(Layer):
    def __init__(self,
                 input_shape,
                 kernel_size=(3, 3, 20),
                 strides=(1, 1),
                 padding='same',
                 initializer='glorot_uniform',
                 rng=None):
        super().__init__()

        if len(input_shape) != 3:
            raise ValueError('Dimension of input_shape must be 3.')

        if len(kernel_size) != 3:
            raise ValueError('Dimension of kernel_size must be 3.')

        if len(strides) != 2:
            raise ValueError('Dimension of strides must be 2.')

        padding = padding.upper()
        if padding not in ('VALID', 'SAME'):
            raise ValueError('padding must be one of \'valid\' or \'same\'.')

        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding

        input_dim = self.input_dim = input_shape
        output_dim = self.output_dim = self._get_output_shape()

        kernel_shape = kernel_size[:2] + (input_shape[2], kernel_size[2])

        self.W = self.kernel_initializer(initializer,
                                         shape=kernel_shape,
                                         name='W')
        self.b = zeros((output_dim), name='b')

    def __repr__(self):
        return '<{}: shape({}, {})>'.format('Dense',
                                            self.input_dim,
                                            self.output_dim)

    @property
    def input_shape(self):
        return self.input_dim

    @property
    def output_shape(self):
        return self.output_dim

    def forward(self, x):
        return tf.nn.conv2d(x, self.W,
                            strides=self.strides,
                            padding=self.padding)

    def _get_output_shape(self):
        input_shape = self.input_shape
        kernel_size = self.kernel_size
        strides = self.strides
        padding = self.padding

        image_size = input_shape[:2]
        channel = kernel_size[2]

        if padding is 'SAME':
            return tuple(list(image_size) + [channel])
        else:
            return tuple(list((np.array(image_size)
                               - np.array(kernel_size[:2]))
                              // np.array(strides) + 1) + [channel])
