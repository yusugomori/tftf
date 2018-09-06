import tensorflow as tf
from .Layer import Layer
from .initializers import normal


class Embedding(Layer):
    def __init__(self, output_dim,
                 input_dim=None,
                 initializer='normal'):
        '''
        # Arguments
            input_dim: num of words (maximum index)
            output_dim: embedding dimension
        '''
        super().__init__()
        self._input_dtype = tf.int32

        self.output_dim = output_dim
        self.input_dim = input_dim
        self.initializer = initializer

        if self.input_dim is None:
            raise ValueError('input_dim must be specified on Embedding layer.')

    def compile(self):
        self.W = \
            self.kernel_initializer(self.initializer,
                                    shape=(self.input_dim, self.output_dim),
                                    name='W')

        self.params = [self.W]

    def forward(self, x, **kwargs):
        return tf.nn.embedding_lookup(self.W, x)
