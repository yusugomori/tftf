import tensorflow as tf
from .Layer import Layer
from .initializers import zeros


class Dense(Layer):
    def __init__(self, input_dim, output_dim,
                 initializer='glorot_normal',
                 rng=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.W = self.kernel_initializer(initializer,
                                         shape=(input_dim, output_dim),
                                         name='W')
        self.b = zeros((output_dim), name='b')

    def forward(self, x):
        return tf.matmul(x, self.W) + self.b
