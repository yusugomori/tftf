import numpy as np
import tensorflow as tf


def ones(shape, name=None, type='float32'):
    init = np.ones(shape).astype(type)
    return tf.Variable(init, name=name)
