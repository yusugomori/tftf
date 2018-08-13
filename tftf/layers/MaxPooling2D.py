import numpy as np
import tensorflow as tf
from .Layer import Layer


class MaxPooling2D(Layer):
    def __init__(self,
                 pool_size=(2, 2),
                 strides=None,
                 padding='valid'):
        super().__init__()

        if len(pool_size) != 2:
            raise ValueError('Dimension of pool_size must be 2.')

        if strides is None:
            strides = pool_size
        elif len(strides) != 2:
            raise ValueError('Dimension of strides must be 2.')

        padding = padding.upper()
        if padding not in ('VALID', 'SAME'):
            raise ValueError('padding must be one of \'valid\' or \'same\'.')

        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding

    @property
    def input_shape(self):
        return self.input_dim

    @property
    def output_shape(self):
        return self.output_dim

    @property
    def _pool_size(self):
        return tuple([1] + list(self.pool_size) + [1])

    @property
    def _strides(self):
        return tuple([1] + list(self.strides) + [1])

    def compile(self):
        pass

    def forward(self, x):
        return tf.nn.max_pool(x,
                              ksize=self._pool_size,
                              strides=self._strides,
                              padding=self.padding)

    def initialize_output_dim(self):
        super().initialize_output_dim()
        self.output_dim = self._get_output_shape()
        return self.output_dim

    def _get_output_shape(self):
        input_shape = self.input_shape
        pool_size = self.pool_size
        strides = self.strides
        padding = self.padding

        if padding == 'SAME':
            return input_shape
        else:
            return tuple(list((np.array(input_shape[:2])
                               - np.array(pool_size))
                              // np.array(strides) + 1) + [input_shape[2]])
