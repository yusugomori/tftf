import tensorflow as tf


def momentum(lr=0.01, momentum=0.9, use_nesterov=True):
    return tf.train.MomentumOptimizer(lr, momentum, use_nesterov)
