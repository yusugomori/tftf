import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tftf.layers import Layer, Dense, Activation, RNN
from tftf.models import Model


if __name__ == '__main__':
    np.random.seed(12345)

    '''
    Load data
    '''
    def sin(x, T=100):
        return np.sin(2.0 * np.pi * x / T)

    def toy_problem(T=100, ampl=0.05):
        x = np.arange(0, 2 * T + 1)
        noise = ampl * np.random.uniform(low=-1.0, high=1.0, size=len(x))
        return sin(x) + noise

    T = 200
    f = toy_problem(T)

    length_of_sequences = 2 * T
    maxlen = 50

    data = []
    target = []

    for i in range(0, length_of_sequences - maxlen + 1):
        data.append(f[i: i + maxlen])
        target.append(f[i + maxlen])

    X = np.array(data, dtype='float32').reshape(len(data), maxlen, 1)
    y = np.array(target, dtype='float32').reshape(len(data), 1)

    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.1)

    '''
    Build model
    '''
    model = Model()
    model.add(RNN(1, 25))
    model.add(Dense(25, 1))
    model.add(Activation('linear'))
    model.compile()

    '''
    Train model
    '''
    model.fit(train_X, train_y, epochs=150, batch_size=50)

    '''
    Test model
    '''
    truncate = maxlen
    Z = X[:1]

    original = [f[i] for i in range(maxlen)]
    predicted = [None for i in range(maxlen)]

    for i in range(length_of_sequences - maxlen + 1):
        _z = Z[-1:]
        _y = model.predict(_z)
        _sequence = np.concatenate((_z.reshape(maxlen, 1)[1:], _y),
                                   axis=0).reshape(1, maxlen, 1)
        Z = np.append(Z, _sequence, axis=0)
        predicted.append(_y.reshape(-1))

    plt.rc('font', family='serif')
    plt.figure()
    plt.ylim([-1.5, 1.5])
    plt.plot(toy_problem(T, ampl=0), linestyle='dotted', color='#aaaaaa')
    plt.plot(original, linestyle='dashed', color='black')
    plt.plot(predicted, color='black')
    plt.show()
