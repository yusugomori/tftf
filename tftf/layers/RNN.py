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
        self.b = zeros((output_dim), name='b')
        self.W_recurrent = \
            self.kernel_initializer(recurrent_initializer,
                                    shape=(output_dim, output_dim),
                                    name='W_recurrent')

    def forward(self, x, **kwargs):
        # TODO: masking padding_value
        def _recurrent(state, elems):
            state = \
                self.recurrent_activation(tf.matmul(elems, self.W)
                                          + tf.matmul(state, self.W_recurrent)
                                          + self.b)
            return state

        initial_state = self._initial_state
        if initial_state is None:
            initial_state = \
                tf.matmul(x[:, 0, :],
                          tf.zeros((self.input_dim, self.output_dim)))

        states = tf.scan(fn=_recurrent,
                         elems=tf.transpose(x, perm=[1, 0, 2]),
                         initializer=initial_state)

        if self._return_sequence is True:
            return tf.transpose(states, perm=[1, 0, 2])
        else:
            return states[-1]
