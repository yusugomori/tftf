import tensorflow as tf
from .Dense import Dense
from .initializers import zeros


class TimeDistributedDense(Dense):
    def __init__(self, output_dim,
                 input_dim=None,
                 initializer='glorot_normal',
                 regularizer=None,
                 rng=None):
        super().__init__(output_dim,
                         input_dim=input_dim,
                         initializer=initializer,
                         regularizer=regularizer,
                         rng=rng)

    def forward(self, x, **kwargs):
        recurr = kwargs['recurrent'] if 'recurrent' in kwargs else True
        if not recurr:
            return tf.matmul(x, self.W) + self.b
        else:
            return tf.einsum('ijk,kl->ijl', x, self.W) + self.b
