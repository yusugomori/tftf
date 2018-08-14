import tensorflow as tf


def l2(alpha=0.):
    reg = L2(alpha)
    return reg.loss


class L2(object):
    def __init__(self, alpha):
        self.alpha = alpha

    def loss(self, weights):
        return self.alpha * tf.nn.l2_loss(weights)
