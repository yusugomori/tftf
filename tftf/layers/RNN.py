import numpy as np
import tensorflow as tf
from .Layer import Layer
from .initializers import zeros


class RNN(Layer):
    def __init__(self, output_dim,
                 input_dim=None,
                 initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 recurrent_activation='tanh',
                 length_of_sequences=None,
                 return_sequence=False,
                 initial_state=None,
                 rng=None):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.initializer = initializer
        self.recurrent_initializer = recurrent_initializer
        self.recurrent_activation = \
            self.activation_initializer(recurrent_activation)
        self._length_of_sequences = length_of_sequences
        self._return_sequence = return_sequence
        self._initial_state = initial_state
        self._use_mask = False

    @property
    def input_shape(self):
        return (self._length_of_sequences, self.input_dim)

    def compile(self):
        input_dim = self.input_dim
        output_dim = self.output_dim
        initializer = self.initializer
        recurrent_initializer = self.recurrent_initializer

        self.W = self.kernel_initializer(initializer,
                                         shape=(input_dim, output_dim),
                                         name='W')
        self.W_recurrent = \
            self.kernel_initializer(recurrent_initializer,
                                    shape=(output_dim, output_dim),
                                    name='W_recurrent')
        self.b = zeros((output_dim), name='b')

        self.params = [self.W, self.W_recurrent, self.b]

    def forward(self, x, **kwargs):
        def _recurrent(state, elems):
            if not self._use_mask:
                x = elems
            else:
                x = elems[0]
                mask = elems[1]
            h = self.recurrent_activation(tf.matmul(x, self.W)
                                          + tf.matmul(state, self.W_recurrent)
                                          + self.b)
            if not self._use_mask:
                return h
            else:
                mask = mask[:, np.newaxis]
                return mask * h + (1 - mask) * state

        initial_state = self._initial_state
        if initial_state is None:
            initial_state = \
                tf.matmul(x[:, 0, :],
                          tf.zeros((self.input_dim, self.output_dim)))

        mask = kwargs['mask'] if 'mask' in kwargs else None
        if mask is None:
            states = tf.scan(fn=_recurrent,
                             elems=tf.transpose(x, perm=[1, 0, 2]),
                             initializer=initial_state)
        else:
            self._use_mask = True
            mask = tf.transpose(mask)
            states = tf.scan(fn=_recurrent,
                             elems=[tf.transpose(x, perm=[1, 0, 2]), mask],
                             initializer=initial_state)

        if self._return_sequence is True:
            return tf.transpose(states, perm=[1, 0, 2])
        else:
            return states[-1]
