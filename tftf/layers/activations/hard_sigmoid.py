import tensorflow as tf


def hard_sigmoid(x):
    return tf.minimum(1.0, tf.maximum(0.0, 0.2 * x + 0.5))
