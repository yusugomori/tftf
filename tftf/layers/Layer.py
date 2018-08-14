from .initializers import *
from ..activations import *


class Layer(object):
    def __init__(self):
        self._input_dim = None
        self._output_dim = None
        self._params = []

    def __repr__(self):
        return '<{}: shape({}, {})>'.format(self.name,
                                            self.input_dim,
                                            self.output_dim)

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
    def output_dim(self):
        return self._output_dim

    @output_dim.setter
    def output_dim(self, val):
        self._output_dim = val

    @property
    def output_shape(self):
        return (self.output_dim,)

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, val):
        if type(val).__name__ != 'list':
            raise AttributeError('type of params must be \'list\', '
                                 'not \'{}\'.'.format(type(val).__name__))
        self._params = val

    def activation_initializer(self, activation):
        activations = {
            'linear': linear,
            'relu': relu,
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
