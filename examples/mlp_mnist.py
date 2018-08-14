import numpy as np
import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tftf.layers import Layer, Dense, Activation, Dropout
from tftf.models import Model


if __name__ == '__main__':
    np.random.seed(0)
    tf.set_random_seed(0)
    '''
    Load data
    '''
    mnist = datasets.fetch_mldata('MNIST original', data_home='.')

    n = len(mnist.data)
    N = 30000
    indices = np.random.permutation(range(n))[:N]

    X = mnist.data[indices]
    X = X / 255.0
    X = X - X.mean(axis=1).reshape(len(X), 1)
    y = mnist.target[indices]
    Y = np.eye(10)[y.astype(int)]

    train_X, test_X, train_y, test_y = train_test_split(X, Y)
    train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y)

    '''
    Build model
    '''
    model = Model()
    model.add(Dense(200,
                    input_dim=784,
                    initializer='glorot_uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(200))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, initializer='glorot_uniform'))
    model.add(Activation('softmax'))
    model.compile()

    # model.describe()
    model.describe_params()

    '''
    Train model
    '''
    model.fit(train_X, train_y,
              validation_data=(valid_X, valid_y),
              metrics=['accuracy', 'f1'],
              epochs=10)

    '''
    Test model
    '''
    print('acc: {:.3}, f1: {:.3}'.format(model.accuracy(test_X, test_y),
                                         model.f1(test_X, test_y)))
