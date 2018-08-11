import tensorflow as tf


def mean_squared_error(y, t):
    loss = tf.reduce_mean(tf.square(y - t))
    return loss
