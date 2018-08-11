import tensorflow as tf


def adadelta(lr=1.0, rho=0.95):
    return tf.train.AdadeltaOptimizer(lr, rho)
