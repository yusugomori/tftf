import tensorflow as tf


class L1(object):
    def __init__(self, alpha):
        self.alpha = alpha

    def loss(self, weights):
        return self.alpha * tf.reduce_sum(tf.abs(weights))


def l1(alpha=0.):
    reg = L1(alpha)
    return reg.loss
