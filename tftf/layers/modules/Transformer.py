import numpy as np
import tensorflow as tf
from . import Module
from .. import Embedding, PositionalEncoding
from .. import LayerNormalization
from .. import TimeDistributedDense as Dense
from .. import Activation, Dropout
from ...losses import categorical_crossentropy
from ...optimizers import adam


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
                 d_ff=2048,
                 N=6,
                 h=8,
                 pad_value=0,
                 p_dropout=0.1,
                 maxlen=6000,
                 warmup_steps=4000):
        self.len_src_vocab = len_src_vocab
        self.len_target_vocab = len_target_vocab
        self.d_model = d_model
        self.d_ff = d_ff
        self.N = N
        self.h = h
        self.pad_value = pad_value
        self.p_dropout = p_dropout
        self.maxlen = maxlen
        self.warmup_steps = warmup_steps
        self.is_training = tf.placeholder_with_default(False, ())

    '''
    Model Architecture
    '''
    def v1(self, x, t, **kwargs):
        mask_src = self._pad_mask(x)
        x = self.encode(x, mask=mask_src, **kwargs)

        mask_tgt = self._pad_subsequent_mask(t)
        x = self.decode(t, memory=x,
                        mask_src=mask_src, mask_tgt=mask_tgt, **kwargs)

        x = Dense(self.len_target_vocab)(x)
        x = Activation('softmax')(x)

        self.x = x
        self.t = tf.one_hot(t, depth=self.len_target_vocab, dtype=tf.float32)

        return x

    def encode(self, x, mask=None, **kwargs):
        x = Embedding(self.d_model, self.len_src_vocab)(x)
        x = PositionalEncoding(self.d_model, self.maxlen)(x)
        x = Dropout(self.p_dropout)(x)

        for n in range(self.N):
            x = self._encoder_sublayer(x, mask=mask)

        return x

    def decode(self, x, memory, mask_src=None, mask_tgt=None, **kwargs):
        x = Embedding(self.d_model, self.len_target_vocab)(x)
        x = PositionalEncoding(self.d_model, self.maxlen)(x)
        x = Dropout(self.p_dropout)(x)

        for n in range(self.N):
            x = self._decoder_sublayer(x, memory,
                                       mask_src=mask_src, mask_tgt=mask_tgt)

        return x

    def _encoder_sublayer(self, x, mask=None):
        # 1st sub-layer
        h = self._multi_head_attention(query=x, key=x, value=x, mask=mask)
        h = Dropout(self.p_dropout)(h)
        x = LayerNormalization()(h + x)

        # 2nd sub-layer
        h = self._feed_forward(x)
        h = Dropout(self.p_dropout)(h)
        x = LayerNormalization()(h + x)

        return x

    def _decoder_sublayer(self, x, memory, mask_src=None, mask_tgt=None):
        # 1st sub-layer
        h = self._multi_head_attention(query=x, key=x, value=x, mask=mask_tgt)
        h = Dropout(self.p_dropout)(h)
        x = LayerNormalization()(h + x)

        # 2nd sub-layer
        h = self._multi_head_attention(query=x, key=memory, value=memory,
                                       mask=mask_src)
        h = Dropout(self.p_dropout)(h)
        x = LayerNormalization()(h + x)

        # 3rd sub-layer
        h = self._feed_forward(x)
        h = Dropout(self.p_dropout)(h)
        x = LayerNormalization()(h + x)

        return x

    def _multi_head_attention(self, query, key, value, mask=None):
        d_k = d_v = self.d_k = self.d_v = self.d_model // self.h
        linears = [Dense(self.d_model, self.d_model) for _ in range(4)]
        n_batches = tf.shape(query)[0]
        query, key, value = \
            [tf.transpose(tf.reshape(l(x),
                                     shape=[n_batches, -1, self.h, d_k]),
                          perm=[0, 2, 1, 3])
             for l, x in zip(linears, (query, key, value))]

        if mask is not None:
            mask = mask[:, np.newaxis]  # apply to all heads
        x, attn = self._attention(query, key, value, mask=mask)
        x = tf.reshape(tf.transpose(x, perm=[0, 2, 1, 3]),
                       shape=[n_batches, -1, self.h * d_k])

        return linears[-1](x)

    def _feed_forward(self, x):
        x = Dense(self.d_ff, self.d_model)(x)
        x = Activation('relu')(x)
        return Dense(self.d_model, self.d_ff)(x)

    def _attention(self, query, key, value, mask=None):
        '''
        Scaled Dot-Product Attention
        '''
        d_k = self.d_k
        score = tf.matmul(query,
                          tf.transpose(key, perm=[0, 1, 3, 2])) / np.sqrt(d_k)
        if mask is not None:
            mask = self._to_attention_mask(mask)
            score *= mask

        attn = tf.nn.softmax(score)
        c = tf.matmul(attn, value)

        return c, attn

    def _pad_mask(self, x):
        mask = tf.cast(tf.not_equal(x, self.pad_value), tf.float32)
        return mask[:, np.newaxis]

    def _subsequent_mask(self, x):
        size = tf.shape(x)[-1]
        shape = (1, size, size)
        mask = tf.matrix_band_part(tf.ones(shape), -1, 0)
        return tf.cast(mask, tf.float32)

    def _pad_subsequent_mask(self, x):
        mask = self._pad_mask(x)
        mask = \
            tf.cast(
                tf.logical_and(tf.cast(mask, tf.bool),
                               tf.cast(self._subsequent_mask(x),
                                       tf.bool)),
                tf.float32
            )
        return mask

    def _to_attention_mask(self, mask):
        return tf.where(condition=tf.equal(mask, 0),
                        x=tf.ones_like(mask,
                                       dtype=tf.float32) * np.float32(-1e+9),
                        y=tf.ones_like(mask,
                                       dtype=tf.float32))

    '''
    Training
    '''
    def loss(self, preds=None, target=None):
        if preds is None:
            preds = self.x
        if target is None:
            target = self.t
        return categorical_crossentropy(preds, target)

    def optimizer(self, loss=None):
        if loss is None:
            loss = self.loss()
        lrate = tf.placeholder(tf.float32, shape=(), name='lrate')
        opt = adam(lr=lrate, beta1=0.9, beta2=0.98, eps=1e-9)
        return (opt.minimize(loss), lrate)

    def lrate(self, epoch=0):
        '''
        Learning rate for Adam in the model
        '''
        step = epoch + 1
        return self.d_model ** (-0.5) * \
            min(step ** (-0.5), step * self.warmup_steps ** (-1.5))
