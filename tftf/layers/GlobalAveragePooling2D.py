import tensorflow as tf
from .Layer import Layer


class GlobalAveragePooling2D(Layer):
    def __init__(self):
        super().__init__()

    def compile(self):
        pass

    def forward(self, x, **kwargs):
        return tf.reduce_mean(x, axis=[1, 2])

    def initialize_output_dim(self):
        super().initialize_output_dim()
        self.output_dim = self.input_dim[-1]
        return self.output_dim
