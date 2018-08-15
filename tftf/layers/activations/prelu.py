import numpy as np
import tensorflow as tf


def prelu(x, type='float32'):
    alpha = tf.Variable(np.zeros([x.get_shape()[-1]]).astype(type),
                        name='alpha')
    return tf.maximum(0., x) + alpha * tf.minimum(0., x)
