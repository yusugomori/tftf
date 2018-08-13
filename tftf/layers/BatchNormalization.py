import numpy as np
import tensorflow as tf
from .Layer import Layer


class BatchNormalization(Layer):
    def __init__(self, input_dim,
                 gamma_initializer='ones',
                 beta_initializer='zeros',
                 eps=np.float32(1e-5)):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = input_dim

        self.gamma = \
            self.kernel_initializer(gamma_initializer,
                                    shape=(input_dim),
                                    name='gamma')
        self.beta = \
            self.kernel_initializer(beta_initializer,
                                    shape=(input_dim),
                                    name='beta')
        self.eps = eps

    def forward(self, x):
        axes = 0
        if len(x.get_shape()) == 4:
            axes = (0, 1, 2)
        mean, var = tf.nn.moments(x, axes, keep_dims=True)
        std = tf.sqrt(var + self.eps)
        return self.gamma * (x - mean) / std + self.beta
