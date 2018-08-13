from .initializers import *
from ..activations import *


class Layer(object):
    def __init__(self):
        self._input_dim = None
        self._output_dim = None

    def __repr__(self):
        return '<{}: shape({}, {})>'.format(self.__class__.__name__,
                                            self.input_dim,
                                            self.output_dim)

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
    def output_dim(self):
        return self._output_dim

    @output_dim.setter
    def output_dim(self, val):
        self._output_dim = val

    @property
    def input_shape(self):
        return (self.input_dim,)

    @property
    def output_shape(self):
        return (self.output_dim,)

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

    def forward(self, x):
        raise NotImplementedError()

    def initialize_output_dim(self):
        if self.input_dim is None:
            raise ValueError('input_dim not definfed.')

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
