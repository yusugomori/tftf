import numpy as np
import tensorflow as tf


def glorot_uniform(shape, name=None, rng=None, type='float32'):
    if rng is None:
        rng = np.random

    high = np.sqrt(6 / (shape[0] + shape[1]))
    init = rng.uniform(low=-high,
                       high=high,
                       size=shape).astype(type)
    return tf.Variable(init, name=name)
