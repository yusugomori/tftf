import numpy as np
import tensorflow as tf


def zeros(shape, name=None, type='float32'):
    init = np.zeros(shape).astype(type)
    return tf.Variable(init, name=name)
