import numpy as np
import tensorflow as tf


def normal(shape, mean=0., std=1., name=None, rng=None, type='float32'):
    if rng is None:
        rng = np.random

    init = rng.normal(loc=std, scale=std, size=shape).astype(type)
    return tf.Variable(init, name=name)
