import tensorflow as tf


def binary_crossentropy(y, t):
    loss = -tf.reduce_mean(t * tf.log(tf.clip_by_value(y, 1e-10, 1.0))
                           + (1. - t) * tf.log(tf.clip_by_value(1. - y,
                                                                1e-10,
                                                                1.0)))
    return loss
