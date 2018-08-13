import numpy as np
import tensorflow as tf
from .Layer import Layer


class Flatten(Layer):
    def __init__(self):
        super().__init__()

    def compile(self):
        pass

    def forward(self, x):
        return tf.reshape(x, (-1, self.output_dim))

    def initialize_output_dim(self):
        super().initialize_output_dim()
        self.output_dim = np.prod(self.input_shape)
        return self.output_dim
