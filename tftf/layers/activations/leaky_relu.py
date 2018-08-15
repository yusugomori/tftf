import tensorflow as tf


def leaky_relu(x, alpha=0.2):
    return tf.nn.leaky_relu(x, alpha=alpha)
