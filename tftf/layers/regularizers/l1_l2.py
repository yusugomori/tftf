import tensorflow as tf
from .l1 import L1
from .l2 import L2


def l1_l2(l1=0., l2=0.):
    reg = L1_L2(l1, l2)
    return reg.loss


class L1_L2(object):
    def __init__(self, l1, l2):
        self.L1 = L1(l1)
        self.L2 = L2(l2)

    def loss(self, weights):
        return self.L1.loss(weights) + self.L2.loss(weights)
