import tensorflow as tf
from .activations import *
from .initializers import *


class Layer(object):
    def __init__(self):
        self._input_dim = None
        self._output_dim = None
        self._input_dtype = tf.float32
        self._output_dtype = tf.float32
        self._params = []
        self._reg_loss = []

    def __repr__(self):
        return '<{}: shape({}, {})>'.format(self.name,
                                            self.input_dim,
                                            self.output_dim)

    def __call__(self, x, **kwargs):
        input_shape = x.get_shape().as_list()
        if len(input_shape) == 2:
            self.input_dim = input_shape[1]
        else:
            self.input_dim = tuple(input_shape[1:])
        self.compile()
        x = self.forward(x)

        output_shape = x.get_shape().as_list()
        if len(output_shape) == 2:
            self.output_dim = output_shape[1]
        else:
            self.output_dim = tuple(output_shape[1:])

        return x

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def shape(self):
        return (self.input_dim, self.output_dim)

    @property
    def input_dim(self):
        return self._input_dim

    @input_dim.setter
    def input_dim(self, val):
        self._input_dim = val

    @property
    def input_shape(self):
        return (self.input_dim,)

    @property
    def input_dtype(self):
        return self._input_dtype

    @property
    def output_dim(self):
        return self._output_dim

    @output_dim.setter
    def output_dim(self, val):
        self._output_dim = val

    @property
    def output_shape(self):
        return (self.output_dim,)

    @property
    def output_dtype(self):
        return self._output_dtype

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, val):
        if type(val).__name__ != 'list':
            raise AttributeError('type of params must be \'list\', '
                                 'not \'{}\'.'.format(type(val).__name__))
        self._params = val

    @property
    def reg_loss(self):
        return self._reg_loss

    @reg_loss.setter
    def reg_loss(self, val):
        if type(val).__name__ != 'list':
            raise AttributeError('type of reg_loss must be \'list\', '
                                 'not \'{}\'.'.format(type(val).__name__))
        self._reg_loss = val

    def activation_initializer(self, activation):
        activations = {
            'elu': elu,
            'hard_sigmoid': hard_sigmoid,
            'leaky_relu': leaky_relu,
            'linear': linear,
            # 'prelu': prelu,
            'relu': relu,
            'selu': selu,
            'sigmoid': sigmoid,
            'softmax': softmax,
            'swish': swish,
            'tanh': tanh
        }
        if activation in activations:
            activation = activations[activation]

        return activation

    def compile(self):
        raise NotImplementedError()

    def forward(self, x, **kwargs):
        raise NotImplementedError()

    def initialize_output_dim(self):
        if self.input_dim is None:
            raise ValueError('input_dim not definfed.')

        self.output_dim = self.input_dim
        return self.output_dim

    def kernel_initializer(self, initializer, shape, name=None):
        initializers = {
            'glorot_normal': glorot_normal,
            'glorot_uniform': glorot_uniform,
            'normal': normal,
            'ones': ones,
            'orthogonal': orthogonal,
            'zeros': zeros
        }

        if initializer in initializers:
            initializer = initializers[initializer]

        return initializer(shape, name=name)
