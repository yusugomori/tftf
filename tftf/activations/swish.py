import tensorflow as tf
from .sigmoid import sigmoid


def swish(x):
    return x * sigmoid(x)
