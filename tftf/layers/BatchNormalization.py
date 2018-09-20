import numpy as np
import tensorflow as tf
from .Layer import Layer


class BatchNormalization(Layer):
    def __init__(self,
                 gamma_initializer='ones',
                 beta_initializer='zeros',
                 eps=np.float32(1e-6)):
        super().__init__()
        self.gamma_initializer = gamma_initializer
        self.beta_initializer = beta_initializer
        self.eps = eps

    def compile(self):
        self.gamma = \
            self.kernel_initializer(self.gamma_initializer,
                                    shape=(self.input_dim),
                                    name='gamma')
        self.beta = \
            self.kernel_initializer(self.beta_initializer,
                                    shape=(self.input_dim),
                                    name='beta')

        self.params = [self.gamma, self.beta]

    def forward(self, x, **kwargs):
        axes = list(range(len(x.get_shape()) - 1))
        mean, var = tf.nn.moments(x, axes, keep_dims=True)
        std = tf.sqrt(var + self.eps)
        return self.gamma * (x - mean) / std + self.beta
