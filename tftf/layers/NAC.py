import tensorflow as tf
from .Layer import Layer
from .initializers import *
from ..activations import sigmoid, tanh


class NAC(Layer):
    '''
    Neural Accumulator
    https://arxiv.org/pdf/1808.00508.pdf
    '''
    def __init__(self, input_dim, output_dim,
                 initializer='normal',
                 rng=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.W_hat = self.kernel_initializer(initializer,
                                             shape=(input_dim, output_dim),
                                             name='W_hat')
        self.M_hat = self.kernel_initializer(initializer,
                                             shape=(input_dim, output_dim),
                                             name='W_hat')
        self.W = tanh(self.W_hat) * sigmoid(self.M_hat)

    def __repr__(self):
        return '<{}: shape({}, {})>'.format('NAC',
                                            self.input_dim,
                                            self.output_dim)

    def forward(self, x):
        return tf.matmul(x, self.W)
