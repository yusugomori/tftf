import tensorflow as tf
from .Layer import Layer
from .initializers import zeros


class LSTM(Layer):
    def __init__(self, input_dim, output_dim,
                 initializer='glorot_uniform',
                 activation='tanh',
                 recurrent_initializer='orthogonal',
                 recurrent_activation='sigmoid',
                 length_of_sequences=None,
                 return_sequence=False,
                 initial_state=None,
                 cell_state=None,
                 rng=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.W_c = \
            self.kernel_initializer(initializer,
                                    shape=(input_dim, output_dim),
                                    name='W_c')
        self.W_i = \
            self.kernel_initializer(initializer,
                                    shape=(input_dim, output_dim),
                                    name='W_i')
        self.W_f = \
            self.kernel_initializer(initializer,
                                    shape=(input_dim, output_dim),
                                    name='W_f')
        self.W_o = \
            self.kernel_initializer(initializer,
                                    shape=(input_dim, output_dim),
                                    name='W_o')
        self.W_recurrent_c = \
            self.kernel_initializer(recurrent_initializer,
                                    shape=(output_dim, output_dim),
                                    name='W_recurrent_c')
        self.W_recurrent_i = \
            self.kernel_initializer(recurrent_initializer,
                                    shape=(output_dim, output_dim),
                                    name='W_recurrent_i')
        self.W_recurrent_f = \
            self.kernel_initializer(recurrent_initializer,
                                    shape=(output_dim, output_dim),
                                    name='W_recurrent_f')
        self.W_recurrent_o = \
            self.kernel_initializer(recurrent_initializer,
                                    shape=(output_dim, output_dim),
                                    name='W_recurrent_o')

        self.b_c = zeros((output_dim), name='b_c')
        self.b_i = zeros((output_dim), name='b_i')
        self.b_f = zeros((output_dim), name='b_f')
        self.b_o = zeros((output_dim), name='b_o')

        self.activation = \
            self.activation_initializer(activation)

        self.recurrent_activation = \
            self.activation_initializer(recurrent_activation)

        self._length_of_sequences = length_of_sequences
        self._return_sequence = return_sequence
        self._initial_state = initial_state
        self._cell_state = cell_state

    @property
    def cell_state(self):
        if self._return_sequence is True:
            return tf.transpose(self._cell_state, perm=[1, 0, 2])
        else:
            return self._cell_state[-1]

    @property
    def input_shape(self):
        return (self._length_of_sequences, self.input_dim)

    def forward(self, x):
        activation = self.activation
        recurrent_activation = self.recurrent_activation

        # TODO: masking padding_value
        def _recurrent(state, elems):
            a = activation(tf.matmul(elems, self.W_c)
                           + tf.matmul(state[0], self.W_recurrent_c)
                           + self.b_c)
            i = recurrent_activation(tf.matmul(elems, self.W_i)
                                     + tf.matmul(state[0], self.W_recurrent_i)
                                     + self.b_i)
            f = recurrent_activation(tf.matmul(elems, self.W_f)
                                     + tf.matmul(state[0], self.W_recurrent_f)
                                     + self.b_f)
            o = recurrent_activation(tf.matmul(elems, self.W_o)
                                     + tf.matmul(state[0], self.W_recurrent_o)
                                     + self.b_o)

            cell = i * a + f * state[1]
            state = o * activation(cell)

            return [state, cell]

        initial_state = self._initial_state
        cell_state = self._cell_state

        if initial_state is None:
            initial_state = \
                tf.matmul(x[:, 0, :],
                          tf.zeros((self.input_dim, self.output_dim)))

        if cell_state is None:
            cell_state = \
                tf.matmul(x[:, 0, :],
                          tf.zeros((self.input_dim, self.output_dim)))

        states, cell = tf.scan(fn=_recurrent,
                               elems=tf.transpose(x, perm=[1, 0, 2]),
                               initializer=[initial_state, cell_state])
        self._cell_state = cell

        if self._return_sequence is True:
            return tf.transpose(states, perm=[1, 0, 2])
        else:
            return states[-1]
