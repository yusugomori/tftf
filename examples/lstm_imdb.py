import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tftf.datasets import load_imdb
from tftf.layers \
    import Dense, Activation, RNN, LSTM, Embedding
from tftf.preprocessing.sequence import Pad
from tftf.preprocessing.sequence import pad_sequences, sort
from tftf.models import Model

if __name__ == '__main__':
    np.random.seed(0)
    tf.set_random_seed(123)

    '''
    Load data
    '''
    num_words = 10000
    imdb = load_imdb(num_words=num_words,
                     train_test_split=False)
    X = imdb.data
    y = imdb.target
    train_X, test_X, train_y, test_y = train_test_split(X, y)
    train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y)

    train_X, train_y = sort(train_X, train_y)
    valid_X, valid_y = sort(valid_X, valid_y)
    test_X, test_y = sort(test_X, test_y)

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

    model.fit(train_X[:10000], train_y[:10000],
              epochs=30,
              shuffle=False,
              metrics=['accuracy', 'f1'],
              preprocesses=[Pad(value=0)],
              validation_data=(valid_X[:5000], valid_y[:5000]))

    '''
    Test model
    '''
    test_X, test_y = test_X[:2000], test_y[:2000]
    test_X = pad_sequences(test_X)
    print(model.accuracy(test_X, test_y))
