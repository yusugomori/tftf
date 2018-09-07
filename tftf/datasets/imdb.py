import numpy as np
import tensorflow as tf
from .Dataset import Dataset


class IMDb(Dataset):
    pass


def load_imdb(one_hot=True,
              num_words=None,
              start_char=1,
              oov_char=2,
              index_from=3,
              train_test_split=True):
    train, valid = \
        tf.keras.datasets.imdb.load_data(num_words=num_words,
                                         start_char=start_char,
                                         oov_char=oov_char,
                                         index_from=index_from)
    train = list(train)
    valid = list(valid)

    if one_hot:
        train[1] = train[1][:, np.newaxis]
        valid[1] = valid[1][:, np.newaxis]

    if not train_test_split:
        data = np.append(train[0], valid[0], axis=0)
        target = np.append(train[1], valid[1], axis=0)

        return IMDb(data, target)

    else:
        return tuple(train), tuple(valid)
