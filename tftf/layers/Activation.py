from .Layer import Layer


class Activation(Layer):
    def __init__(self, activation='linear'):
        super().__init__()
        self.activation = self.activation_initializer(activation)

    def __repr__(self):
        return '<{}: {}({}, {})>'.format(self.__class__.__name__,
                                         self.activation.__name__,
                                         self.input_dim,
                                         self.output_dim)

    def compile(self):
        pass

    def forward(self, x):
        return self.activation(x)

    def initialize_output_dim(self):
        super().initialize_output_dim()
        self.output_dim = self.input_dim
        return self.output_dim
