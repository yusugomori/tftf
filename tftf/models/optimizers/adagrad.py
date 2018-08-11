import tensorflow as tf


def adagrad(lr=0.01):
    return tf.train.AdagradOptimizer(lr)
