import tensorflow as tf
from .NAC import NAC
from .initializers import *
from ..activations import sigmoid


class NALU(NAC):
    '''
    Neural Arithmetic Logic Unit
    https://arxiv.org/pdf/1808.00508.pdf
    '''
    def __init__(self, input_dim, output_dim,
                 initializer='normal',
                 rng=None):
        super().__init__(input_dim, output_dim, initializer, rng)

        self.G = self.kernel_initializer(initializer,
                                         shape=(input_dim, output_dim),
                                         name='G')

    def __repr__(self):
        return '<{}: shape({}, {})>'.format('NALU',
                                            self.input_dim,
                                            self.output_dim)

    def forward(self, x):
        eps = 1e-8
        self.g = sigmoid(tf.matmul(x, self.G))
        self.m = tf.exp(tf.matmul(tf.log(tf.abs(x) + eps), self.W))
        self.a = tf.matmul(x, self.W)

        return self.g * self.a + (1 - self.g) * self.m
