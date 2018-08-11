from .initializers import *


class Layer(object):
    def __init__(self):
        self.input_dim = None
        self.output_dim = None

    def forward(self, x):
        raise NotImplementedError()

    def kernel_initializer(self, initializer, shape, name=None):
        initializers = {
            'glorot_normal': glorot_normal,
            'glorot_uniform': glorot_uniform,
            'normal': normal
        }

        if initializer in initializers:
            initializer = initializers[initializer]

        return initializer(shape, name=name)
