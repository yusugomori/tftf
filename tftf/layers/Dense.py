import tensorflow as tf
from .Layer import Layer
from .initializers import zeros


class Dense(Layer):
    def __init__(self, output_dim,
                 input_dim=None,
                 initializer='glorot_normal',
                 rng=None):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.initializer = initializer

    def compile(self):
        self.W = \
            self.kernel_initializer(self.initializer,
                                    shape=(self.input_dim, self.output_dim),
                                    name='W')
        self.b = zeros((self.output_dim), name='b')

    def forward(self, x, **kwargs):
        return tf.matmul(x, self.W) + self.b
