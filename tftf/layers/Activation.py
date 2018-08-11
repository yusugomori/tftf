from .Layer import Layer
from ..activations import *


class Activation(Layer):
    def __init__(self, activation='linear'):
        super().__init__()
        self.input_dim = None
        self.output_dim = None
        activations = {
            'linear': linear,
            'relu': relu,
            'sigmoid': sigmoid,
            'softmax': softmax,
            'swish': swish,
            'tanh': tanh
        }

        if activation in activations:
            self.activation = activations[activation]
        else:
            self.activation = activation

    def __repr__(self):
        return '<{}: {}({}, {})>'.format('Activation',
                                         self.activation.__name__,
                                         self.input_dim,
                                         self.output_dim)

    def forward(self, x):
        return self.activation(x)
