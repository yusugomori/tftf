import tensorflow as tf


def rmsprop(lr=0.001):
    return tf.train.RMSPropOptimizer(lr)
