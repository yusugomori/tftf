import numpy as np
import tensorflow as tf
from .Dataset import Dataset


class MNIST(Dataset):
    pass


def load_mnist(one_hot=True,
               train_test_split=True,
               flatten=False,
               include_channel=True):

    train, valid = tf.keras.datasets.mnist.load_data()
    train = list(train)
    valid = list(valid)

    if flatten:
        train[0] = train[0].reshape(-1, 784)
        valid[0] = valid[0].reshape(-1, 784)
    elif include_channel:
        train[0] = train[0].reshape(len(train[0]), 28, 28, 1)
        valid[0] = valid[0].reshape(len(valid[0]), 28, 28, 1)

    if one_hot:
        train[1] = np.eye(10)[train[1].astype(int)]
        valid[1] = np.eye(10)[valid[1].astype(int)]

    if not train_test_split:
        data = np.append(train[0], valid[0], axis=0)
        target = np.append(train[1], valid[1], axis=0)
        return MNIST(data, target)
    else:
        return tuple(train), tuple(valid)
