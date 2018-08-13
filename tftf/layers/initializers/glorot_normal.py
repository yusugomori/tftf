import numpy as np
import tensorflow as tf


def glorot_normal(shape, name=None, rng=None, type='float32'):
    if rng is None:
        rng = np.random

    if len(shape) == 2:
        fan_in = shape[0]
    elif len(shape) == 4:
        fan_in = np.prod(shape[:3])
    else:
        raise ValueError('Dimension of shape must be 2 or 4.')

    init = np.sqrt(1 / fan_in) * rng.normal(size=shape).astype(type)
    return tf.Variable(init, name=name)
