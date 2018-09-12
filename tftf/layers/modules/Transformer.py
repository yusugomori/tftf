import numpy as np
import tensorflow as tf
from . import Module
from .. import Embedding, PositionalEncoding
from .. import Dropout


class Transformer(Module):
    '''
    Implementation of Transformer model from
    "Attention Is All You Need",
    Ashish Vaswani et al.
    https://arxiv.org/abs/1706.03762
    '''
    def __init__(self,
                 len_src_vocab,
                 len_target_vocab,
                 d_model=512,
                 N=6,
                 maxlen=6000):
        self.len_src_vocab = len_src_vocab
        self.len_target_vocab = len_target_vocab
        self.d_model = d_model
        self.N = N
        self.maxlen = maxlen

        self._initialize_positional_encoding()

    def v1(self, x, **kwargs):
        x = Embedding(self.d_model, self.len_src_vocab)(x) \
            + PositionalEncoding(self.d_model, self.maxlen)(x)
        pass

    def embedding(self):
        pass

    def scaled_dot_product_attention(self, Q, K, V):
        d_k = K.get_shape()[0]
        return tf.dot(tf.softmax(tf.dot(Q, tf.sqrt(K.T)) / d_k), V)

    def multi_head_attention(self):
        pass

    def masked_multi_head_attention(self):
        pass

    def add(self):
        pass

    def norm(self):
        pass

    def feed_forward(self):
        pass
