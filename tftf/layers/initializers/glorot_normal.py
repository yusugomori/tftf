import numpy as np
import tensorflow as tf


def glorot_normal(shape, name=None, rng=None, type='float32'):
    if rng is None:
        rng = np.random

    init = np.sqrt(1 / shape[0]) * rng.normal(size=shape).astype(type)
    return tf.Variable(init, name=name)
