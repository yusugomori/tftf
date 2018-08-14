import numpy as np
import tensorflow as tf
from .Layer import Layer


class Conv2D(Layer):
    def __init__(self,
                 input_dim=None,
                 kernel_size=(3, 3, 20),
                 strides=(1, 1),
                 padding='same',
                 initializer='glorot_uniform',
                 rng=None):
        super().__init__()

        if input_dim is not None and len(input_dim) != 3:
            raise ValueError('Dimension of input_dim must be 3.')

        if len(kernel_size) != 3:
            raise ValueError('Dimension of kernel_size must be 3.')

        if len(strides) != 2:
            raise ValueError('Dimension of strides must be 2.')

        padding = padding.upper()
        if padding not in ('VALID', 'SAME'):
            raise ValueError('padding must be one of \'valid\' or \'same\'.')

        self.input_dim = input_dim
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.initializer = initializer

    @property
    def input_shape(self):
        return self.input_dim

    @property
    def output_shape(self):
        return self.output_dim

    @property
    def _strides(self):
        return tuple([1] + list(self.strides) + [1])

    def compile(self):
        kernel_shape = \
            self.kernel_size[:2] + (self.input_dim[2], self.kernel_size[2])

        self.W = self.kernel_initializer(self.initializer,
                                         shape=kernel_shape,
                                         name='W')

        self.params = [self.W]

    def forward(self, x, **kwargs):
        return tf.nn.conv2d(x, self.W,
                            strides=self._strides,
                            padding=self.padding)

    def initialize_output_dim(self):
        super().initialize_output_dim()
        self.output_dim = self._get_output_shape()
        return self.output_dim

    def _get_output_shape(self):
        input_shape = self.input_shape
        kernel_size = self.kernel_size
        strides = self.strides
        padding = self.padding

        image_size = input_shape[:2]
        channel = kernel_size[2]

        if padding == 'SAME':
            return tuple(list(image_size) + [channel])
        else:
            return tuple(list((np.array(image_size)
                               - np.array(kernel_size[:2]))
                              // np.array(strides) + 1) + [channel])
