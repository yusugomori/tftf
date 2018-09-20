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
                 label_smooth=0.1,
                 maxlen=6000,
                 warmup_steps=4000):

        assert label_smooth >= 0.

        self.len_src_vocab = len_src_vocab
        self.len_target_vocab = len_target_vocab
        self.d_model = d_model
        self.d_ff = d_ff
        self.N = N
        self.h = h
        self.pad_value = pad_value
        self.p_dropout = p_dropout
        self.label_smooth = label_smooth
        self.maxlen = maxlen
        self.warmup_steps = warmup_steps
        self.is_training = tf.placeholder_with_default(False, ())

    '''
    Model Architecture
    '''
    def v1(self, x, t, **kwargs):
        mask_src = self._pad_mask(x)
        encoder = self.encoder = \
            Encoder(len_src_vocab=self.len_src_vocab,
                    N=self.N,
                    d_model=self.d_model,
                    d_ff=self.d_ff,
                    h=self.h,
                    p_dropout=self.p_dropout,
                    maxlen=self.maxlen)

        x = self.encode(x, mask=mask_src)

        mask_tgt = self._pad_subsequent_mask(t)
        decoder = self.decoder = \
            Decoder(len_target_vocab=self.len_target_vocab,
                    N=self.N,
                    d_model=self.d_model,
                    d_ff=self.d_ff,
                    h=self.h,
                    p_dropout=self.p_dropout,
                    maxlen=self.maxlen)

        x = self.decode(t, memory=x,
                        mask_src=mask_src, mask_tgt=mask_tgt, **kwargs)

        generator = self.generator = \
            Generator(len_target_vocab=self.len_target_vocab)
        x = self.generate(x, **kwargs)

        self.x = x
        self.t = tf.one_hot(t, depth=self.len_target_vocab, dtype=tf.float32)

        return x

    def encode(self, x, mask=None, **kwargs):
        return self.encoder(x, mask=mask,
                            training=self.is_training, **kwargs)

    def decode(self, x, memory, mask_src=None, mask_tgt=None, **kwargs):
        return self.decoder(x, memory=memory,
                            mask_src=mask_src, mask_tgt=mask_tgt,
                            training=self.is_training, **kwargs)

    def generate(self, x, **kwargs):
        return self.generator(x, training=self.is_training, **kwargs)

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

    '''
    Training
    '''
    def loss(self, preds=None, target=None):
        if preds is None:
            preds = self.x
        if target is None:
            target = self.t

        e = self.label_smooth
        if e > 0.:
            target = (1 - e) * target + e / self.len_target_vocab
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


class Encoder(object):
    def __init__(self,
                 len_src_vocab,
                 N=6,
                 d_model=512,
                 d_ff=2048,
                 h=8,
                 p_dropout=0.1,
                 maxlen=6000):
        self.layers = [
            Embedding(d_model, len_src_vocab),
            PositionalEncoding(d_model, maxlen),
            Dropout(p_dropout)
        ]
        self.sub_layers = \
            [EncoderSubLayer(d_model, d_ff, h, p_dropout) for _ in range(N)]

    def __call__(self, x, mask=None, **kwargs):
        return self.forward(x, mask, **kwargs)

    def forward(self, x, mask=None, **kwargs):
        for l in self.layers:
            x = l(x, **kwargs)
        for sub_layer in self.sub_layers:
            x = sub_layer(x, mask=mask, **kwargs)

        return x


class EncoderSubLayer(object):
    def __init__(self,
                 d_model,
                 d_ff,
                 h,
                 p_dropout):
        self.layers = [
            [MultiHeadAttention(d_model, h),
             Dropout(p_dropout),
             LayerNormalization()],
            [FFN(d_model, d_ff),
             Dropout(p_dropout),
             LayerNormalization()]
        ]

    def __call__(self, x, mask=None, **kwargs):
        return self.forward(x, mask, **kwargs)

    def forward(self, x, mask=None, **kwargs):
        # 1st sub-layer
        layers = self.layers[0]
        h = layers[0](query=x, key=x, value=x, mask=mask, **kwargs)
        h = layers[1](h, **kwargs)
        x = layers[2](h + x, **kwargs)

        # 2nd sub-layer
        layers = self.layers[1]
        h = layers[0](x, **kwargs)
        h = layers[1](h, **kwargs)
        x = layers[2](h + x, **kwargs)

        return x


class Decoder(object):
    def __init__(self,
                 len_target_vocab,
                 N=6,
                 d_model=512,
                 d_ff=2048,
                 h=8,
                 p_dropout=0.1,
                 maxlen=6000):
        self.layers = [
            Embedding(d_model, len_target_vocab),
            PositionalEncoding(d_model, maxlen),
            Dropout(p_dropout)
        ]
        self.sub_layers = \
            [DecoderSubLayer(d_model, d_ff, h, p_dropout) for _ in range(N)]

    def __call__(self, x, memory,
                 mask_src=None, mask_tgt=None, **kwargs):
        return self.forward(x, memory, mask_src, mask_tgt, **kwargs)

    def forward(self, x, memory,
                mask_src=None, mask_tgt=None, **kwargs):
        for l in self.layers:
            x = l(x, **kwargs)
        for sub_layer in self.sub_layers:
            x = sub_layer(x, memory,
                          mask_src=mask_src, mask_tgt=mask_tgt, **kwargs)

        return x


class DecoderSubLayer(object):
    def __init__(self,
                 d_model,
                 d_ff,
                 h,
                 p_dropout):
        self.layers = [
            [MultiHeadAttention(d_model, h),
             Dropout(p_dropout),
             LayerNormalization()],
            [MultiHeadAttention(d_model, h),
             Dropout(p_dropout),
             LayerNormalization()],
            [FFN(d_model, d_ff),
             Dropout(p_dropout),
             LayerNormalization()]
        ]

    def __call__(self, x, memory,
                 mask_src=None, mask_tgt=None, **kwargs):
        return self.forward(x, memory, mask_src, mask_tgt, **kwargs)

    def forward(self, x, memory,
                mask_src=None, mask_tgt=None, **kwargs):
        # 1st sub_layer
        layers = self.layers[0]
        h = layers[0](query=x, key=x, value=x, mask=mask_tgt, **kwargs)
        h = layers[1](h, **kwargs)
        x = layers[2](h + x, **kwargs)

        # 2nd sub-layer
        layers = self.layers[1]
        h = layers[0](query=x, key=memory, value=memory,
                      mask=mask_src, **kwargs)
        h = layers[1](h, **kwargs)
        x = layers[2](h + x, **kwargs)

        # 3rd sub-layer
        layers = self.layers[2]
        h = layers[0](x, **kwargs)
        h = layers[1](h, **kwargs)
        x = layers[2](h + x, **kwargs)

        return x


class Generator(object):
    def __init__(self,
                 len_target_vocab):
        self.layers = [
            Dense(len_target_vocab),
            Activation('softmax')
        ]

    def __call__(self, x, **kwargs):
        return self.forward(x, **kwargs)

    def forward(self, x, **kwargs):
        for l in self.layers:
            x = l(x, **kwargs)
        return x


class MultiHeadAttention(object):
    '''
    Multi-Head Attention / Masked Multi-Head Attention
    '''
    def __init__(self, d_model, h):
        self.d_model = d_model
        self.h = h
        self.linears = [Dense(d_model, d_model) for _ in range(4)]

    def __call__(self, query, key, value, mask=None, **kwargs):
        return self.forward(query, key, value, mask, **kwargs)

    def forward(self, query, key, value, mask=None, **kwargs):
        d_k = d_v = self.d_k = self.d_v = self.d_model // self.h
        n_batches = tf.shape(query)[0]
        query, key, value = \
            [tf.transpose(tf.reshape(l(x),
                                     shape=[n_batches, -1, self.h, d_k]),
                          perm=[0, 2, 1, 3])
             for l, x in zip(self.linears, (query, key, value))]

        if mask is not None:
            mask = mask[:, np.newaxis]  # apply to all heads
        x, attn = self._attention(query, key, value, mask=mask, **kwargs)
        x = tf.reshape(tf.transpose(x, perm=[0, 2, 1, 3]),
                       shape=[n_batches, -1, self.h * d_k])

        return self.linears[-1](x)

    def _attention(self, query, key, value, mask=None, **kwargs):
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

    def _to_attention_mask(self, mask):
        return tf.where(condition=tf.equal(mask, 0),
                        x=tf.ones_like(mask,
                                       dtype=tf.float32) * np.float32(-1e+9),
                        y=tf.ones_like(mask,
                                       dtype=tf.float32))


class FFN(object):
    '''
    Position-wise Feed-Forward Networks
    '''
    def __init__(self, d_model, d_ff):
        self.layers = [
            Dense(d_ff, d_model),
            Activation('relu'),
            Dense(d_model, d_ff)
        ]

    def __call__(self, x, **kwargs):
        return self.forward(x, **kwargs)

    def forward(self, x, **kwargs):
        for l in self.layers:
            x = l(x, **kwargs)
        return x
