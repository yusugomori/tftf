import tensorflow as tf


def adam(lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
    return tf.train.AdamOptimizer(lr, beta1, beta2, eps)
