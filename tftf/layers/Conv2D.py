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

        padding = padding.upper()

        self.input_dim = None
        self.output_dim = None
        kernel_shape = kernel_size[:2] + (input_shape[2], kernel_size[2])

        self.W = self.kernel_initializer(initializer,
                                         shape=kernel_shape,
                                         name='W')
        self.b = zeros((output_dim), name='b')

        self.strides = strides
        self.padding = padding

    def __repr__(self):
        return '<{}: shape({}, {})>'.format('Dense',
                                            self.input_dim,
                                            self.output_dim)

    def forward(self, x):
        return tf.nn.conv2d(x, self.W,
                            strides=self.strides,
                            padding=self.padding)
