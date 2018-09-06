import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tftf.layers \
    import Dense, Activation, RNN, LSTM, Embedding
from tftf.preprocessing.sequence import Pad
from tftf.models import Model
from tftf.layers.modules import ResNet
from tftf.preprocessing.sequence import pad_sequences, sort

if __name__ == '__main__':
    np.random.seed(0)
    tf.set_random_seed(123)

    '''
    Load data
    '''
    num_words = 10000
    (train_X, train_y), (test_X, test_y) = \
        tf.keras.datasets.imdb.load_data(num_words=num_words)

    train_X, valid_X, train_y, valid_y = \
        train_test_split(train_X, train_y)

    train_X, train_y = sort(train_X, train_y, order='descend')
    train_X = np.array(train_X)
    train_y = np.array(train_y)[:, np.newaxis]
    valid_X = np.array(valid_X)
    valid_y = np.array(valid_y)[:, np.newaxis]

    '''
    Build model
    '''
    model = Model()
    model.add(Embedding(100, input_dim=num_words))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(variable_input=True,
                  use_mask=True,
                  pad_value=0)
    model.describe()

    '''
    Train model
    '''
    model.fit(train_X, train_y,
              epochs=30,
              metrics=['accuracy', 'f1'],
              preprocesses=[Pad(value=0)],
              validation_data=(valid_X, valid_y))
