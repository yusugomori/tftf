import tensorflow as tf
from .Layer import Layer
from .initializers import *


class RNN(Layer):
    def __init__(self, input_dim, output_dim,
                 initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 recurrent_activation='tanh',
                 length_of_sequences=None,
                 rng=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.W = self.kernel_initializer(initializer,
                                         shape=(input_dim, output_dim),
                                         name='W')
        self.b = zeros((output_dim), name='b')
        self.W_recurrent = \
            self.kernel_initializer(recurrent_initializer,
                                    shape=(output_dim, output_dim),
                                    name='W_recurrent')
        self.recurrent_activation = \
            self.activation_initializer(recurrent_activation)

        self._length_of_sequences = length_of_sequences

    def __repr__(self):
        return '<{}: shape({}, {})>'.format('RNN',
                                            self.input_dim,
                                            self.output_dim)

    @property
    def input_shape(self):
        return (self._length_of_sequences, self.input_dim)

    def forward(self, x, masking=True, padding_value=-1.):
        def _recurrent(state, elems):
            state = \
                self.recurrent_activation(tf.matmul(elems, self.W)
                                          + tf.matmul(state, self.W_recurrent)
                                          + self.b)
            return state

        # if masking is True:
        #     mask = tf.cast(tf.not_equal(x, padding_value), tf.float32)
        # _x =

        initial_state = \
            tf.matmul(x[:, 0, :],
                      tf.zeros((self.input_dim, self.output_dim)))
        states = tf.scan(fn=_recurrent,
                         elems=tf.transpose(x, perm=[1, 0, 2]),
                         initializer=initial_state)

        return states[-1]
