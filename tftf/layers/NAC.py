import tensorflow as tf
from .Layer import Layer
from ..activations import sigmoid, tanh


class NAC(Layer):
    '''
    Neural Accumulator
    https://arxiv.org/pdf/1808.00508.pdf
    '''
    def __init__(self, output_dim,
                 input_dim=None,
                 initializer='normal',
                 rng=None):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.initializer = initializer

    def compile(self):
        self.W_hat = \
            self.kernel_initializer(self.initializer,
                                    shape=(self.input_dim, self.output_dim),
                                    name='W_hat')
        self.M_hat = \
            self.kernel_initializer(self.initializer,
                                    shape=(self.input_dim, self.output_dim),
                                    name='W_hat')
        self.W = tanh(self.W_hat) * sigmoid(self.M_hat)

    def forward(self, x):
        return tf.matmul(x, self.W)
