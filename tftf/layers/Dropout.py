import tensorflow as tf
from .Layer import Layer


class Dropout(Layer):
    def __init__(self, p_dropout, rng=None):
        super().__init__()
        if p_dropout < 0. or p_dropout >= 1:
            raise ValueError('p_dropout must be between 0 and 1.')
        self.p = 1. - p_dropout

    def compile(self):
        pass

    def forward(self, x, **kwargs):
        training = kwargs['training'] \
            if 'training' in kwargs else tf.constant(False, dtype=tf.bool)
        p = tf.cond(training, lambda: self.p, lambda: 1.)
        return tf.nn.dropout(x, p)
