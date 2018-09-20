import numpy as np
import tensorflow as tf
from .Layer import Layer


class LayerNormalization(Layer):
    def __init__(self,
                 a_initializer='ones',
                 b_initializer='zeros',
                 eps=np.float32(1e-6)):
        super().__init__()
        self.a_initializer = a_initializer
        self.b_initializer = b_initializer
        self.eps = eps

    def compile(self):
        self.a = \
            self.kernel_initializer(self.a_initializer,
                                    shape=(self.input_dim),
                                    name='a')
        self.b = \
            self.kernel_initializer(self.b_initializer,
                                    shape=(self.input_dim),
                                    name='b')

        self.params = [self.a, self.b]

    def forward(self, x, **kwargs):
        axes = 0
        mean, var = tf.nn.moments(x, axes=-1, keep_dims=True)
        std = tf.sqrt(var) + self.eps
        return self.a * (x - mean) / std + self.b
