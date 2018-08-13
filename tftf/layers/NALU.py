import tensorflow as tf
from .NAC import NAC
from ..activations import sigmoid


class NALU(NAC):
    '''
    Neural Arithmetic Logic Unit
    https://arxiv.org/pdf/1808.00508.pdf
    '''
    def __init__(self, output_dim,
                 input_dim=None,
                 initializer='normal',
                 rng=None):
        super().__init__(output_dim,
                         input_dim=input_dim,
                         initializer=initializer,
                         rng=rng)

    def compile(self):
        super().compile()
        self.G = \
            self.kernel_initializer(self.initializer,
                                    shape=(self.input_dim, self.output_dim),
                                    name='G')

    def forward(self, x, **kwargs):
        eps = 1e-8
        self.g = sigmoid(tf.matmul(x, self.G))
        self.m = tf.exp(tf.matmul(tf.log(tf.abs(x) + eps), self.W))
        self.a = tf.matmul(x, self.W)

        return self.g * self.a + (1 - self.g) * self.m
