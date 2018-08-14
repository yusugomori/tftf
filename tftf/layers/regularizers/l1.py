import tensorflow as tf


def l1(alpha=0.):
    reg = L1(alpha)
    return reg.loss


class L1(object):
    def __init__(self, alpha):
        self.alpha = alpha

    def loss(self, weights):
        return self.alpha * tf.reduce_sum(tf.abs(weights))
