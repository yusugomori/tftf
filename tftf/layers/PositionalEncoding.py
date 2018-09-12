import numpy as np
import tensorflow as tf
from .Layer import Layer


class PositionalEncoding(Layer):
    def __init__(self, output_dim,
                 maxlen=6000):
        '''
        Positional encoding layer with sinusoid

        # Arguments
            maxlen: max length of sequence
        '''
        super().__init__()
        self.output_dim = output_dim
        self.maxlen = maxlen
        self._pe = self._initialize_pe()

    def compile(self):
        pass

    def forward(self, x, **kwargs):
        return tf.cast(x, tf.float32) \
            + self._pe[:, :x.get_shape().as_list()[1]]

    def _initialize_pe(self):
        pe = np.zeros(shape=(self.maxlen, self.output_dim), dtype=np.float32)
        pos = np.arange(0, self.maxlen)[:, np.newaxis]
        div = np.exp(np.arange(0, self.output_dim, 2)
                     * -(np.log(10000.0) / self.output_dim))

        pe[:, 0::2] = np.sin(pos * div)
        pe[:, 1::2] = np.cos(pos * div)
        return tf.constant(pe[np.newaxis, :])
