import numpy as np
import tensorflow as tf
from .Layer import Layer
from .initializers import zeros


class Attention(Layer):
    '''
    Attention Layer for Seq2Seq
    "Effective Approaches to Attention-based Neural Machine Translation",
    Minh-Thang Luong et al., EMNLP 2015
    https://arxiv.org/abs/1508.04025
    '''
    def __init__(self, output_dim,
                 input_dim=None,
                 initializer='glorot_uniform',
                 activation='tanh',
                 state=None):
        '''
        # Arguments
            input_dim: tuple or list. shape of (encoder_dim, decoder_dim).
            state: (default None). Encoder state (output).
                   shape of (batch_size, len_sequence, encoder_dim)
        '''
        super().__init__()

        if type(input_dim) != list and type(input_dim) != tuple:
            raise ValueError('`input_dim` must be given as a list or tuple.')

        if len(input_dim) != 2:
            raise ValueError('Length of `input_dim` must be 2. '
                             'Not {}.'.format(len(input_dim)))

        if state is None:
            raise ValueError('`state` must be given.')

        self.output_dim = output_dim
        self.input_dim = input_dim

        self.initializer = initializer
        self.activation = \
            self.activation_initializer(activation)
        self.state = state
        self._use_mask = False
        self.mask = None

    @property
    def input_shape(self):
        return tuple(self.input_dim)

    def compile(self):
        input_dim = self.input_dim
        output_dim = self.output_dim
        initializer = self.initializer

        self.W_a = \
            self.kernel_initializer(initializer,
                                    shape=(input_dim[0], input_dim[1]),
                                    name='W_a')
        self.W_c = \
            self.kernel_initializer(initializer,
                                    shape=(input_dim[0], output_dim),
                                    name='W_c')
        self.W_h = \
            self.kernel_initializer(initializer,
                                    shape=(input_dim[1], output_dim),
                                    name='W_h')
        self.b = zeros((output_dim), name='b')

        self.params = [self.W_a, self.W_c, self.W_h, self.b]

    def forward(self, x, **kwargs):
        '''
        # Arguments
            mask: Tensor. Mask for padded value.
                  shape of (batch_size, encoder_dim)
            recurrent: boolean (default True).
        '''
        if self.mask is None:
            self.mask = kwargs['mask'] if 'mask' in kwargs else None
        self._use_mask = True if self.mask is not None else False

        recurr = kwargs['recurrent'] if 'recurrent' in kwargs else True

        if recurr:
            score = tf.einsum('ijk,ilk->ijl',
                              x,
                              tf.einsum('ijk,kl->ijl', self.state, self.W_a))
            if self._use_mask:
                score *= self.mask[:, np.newaxis]

            attn = self.attn = tf.nn.softmax(score)
            c = tf.einsum('ijk,ikl->ijl', attn, self.state)

            return self.activation(tf.einsum('ijk,kl->ijl', c, self.W_c)
                                   + tf.einsum('ijk,kl->ijl', x, self.W_h)
                                   + self.b)
        else:
            score = tf.einsum('ij,ikj->ik',
                              x,
                              tf.einsum('ijk,kl->ijl', self.state, self.W_a))
            if self._use_mask:
                score *= self.mask

            attn = self.attn = tf.nn.softmax(score)
            c = tf.einsum('ij,ijk->ik', attn, self.state)

            return self.activation(tf.matmul(c, self.W_c)
                                   + tf.matmul(x, self.W_h)
                                   + self.b)
