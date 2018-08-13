import numpy as np
import tensorflow as tf


def glorot_uniform(shape, name=None, rng=None, type='float32'):
    if rng is None:
        rng = np.random

    if len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    elif len(shape) == 4:
        fan_in = np.prod(shape[:3])
        fan_out = np.prod(shape[:2]) * shape[3]
    else:
        raise ValueError('Dimension of shape must be 2 or 4.')

    high = np.sqrt(6 / (fan_in + fan_out))
    init = rng.uniform(low=-high,
                       high=high,
                       size=shape).astype(type)
    return tf.Variable(init, name=name)
