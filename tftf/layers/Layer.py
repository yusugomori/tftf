from .initializers import *
from ..activations import *


class Layer(object):
    def __init__(self):
        self.input_dim = None
        self.output_dim = None

    @property
    def shape(self):
        return (self.input_dim, self.output_dim)

    @property
    def input_shape(self):
        return (self.input_dim,)

    @property
    def output_shape(self):
        return (self.output_dim,)

    def forward(self, x):
        raise NotImplementedError()

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

    def kernel_initializer(self, initializer, shape, name=None):
        initializers = {
            'glorot_normal': glorot_normal,
            'glorot_uniform': glorot_uniform,
            'orthogonal': orthogonal,
            'normal': normal,
            'zeros': zeros
        }

        if initializer in initializers:
            initializer = initializers[initializer]

        return initializer(shape, name=name)
