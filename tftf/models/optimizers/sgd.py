import tensorflow as tf


def sgd(lr=0.01):
    return tf.train.GradientDescentOptimizer(lr)
