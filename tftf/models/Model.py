import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from ..layers import Dense
from .losses import *
from .optimizers import *


class Model(object):
    def __init__(self):
        self._layers = []
        self._shapes = []

    # def __del__(self):
    #     if self._sess is not None:
    #         self._sess.close()

    @property
    def layers(self):
        return self._layers

    def add(self, layer):
        input_dim = layer.input_dim
        output_dim = layer.output_dim

        if len(self.layers) == 0:
            if input_dim is None:
                raise AttributeError('input_dim must be specified \
                                      on first layer.')
        else:
            if input_dim is None:
                input_dim = layer.input_dim = self._shapes[-1][1]
            if output_dim is None:
                output_dim = layer.output_dim = input_dim

        self._shapes.append((input_dim, output_dim))
        self._layers.append(layer)

    def compile(self, loss='mse', optimizer='rmsprop'):
        self._set_loss_function(loss)
        self._set_optimize(optimizer)

        input_shape = [None] + list(self.layers[0].input_shape)
        output_shape = [None] + list(self.layers[-1].output_shape)

        x = self.data = tf.placeholder(tf.float32, shape=input_shape)
        t = self.target = tf.placeholder(tf.float32, shape=output_shape)
        y = self._y = self._predict(x)
        self._loss = self._loss_function(y, t)
        self._set_accuracy(y, t)
        self._train_step = self._optimize().minimize(self._loss)
        self._init = tf.global_variables_initializer()
        self._sess = tf.Session()
        self._sess.run(self._init)

    # TODO: validation data
    def fit(self, data, target, epochs=10, batch_size=100):
        if len(data) != len(target):
            raise AttributeError('Length of X and y does not match.')
        n_data = len(data)
        n_batches = n_data // batch_size

        for epoch in range(epochs):
            indices = shuffle(np.arange(n_data))
            _data = data[indices]
            _target = target[indices]

            for i in range(n_batches):
                _start = i * batch_size
                _end = _start + batch_size

                self._sess.run(self._train_step,
                               feed_dict={
                                   self.data: _data[_start:_end],
                                   self.target: _target[_start:_end]
                               })
            _loss = self.loss(_data, _target)
            _acc = self.accuracy(_data, _target)
            print('Epoch: {}, loss: {:.3}, acc: {:.3}'
                  .format((epoch + 1), _loss, _acc))

    def predict(self, data):
        ret = self._sess.run(self._y,
                             feed_dict={
                                 self.data: data
                             })
        return ret

    def loss(self, data, target):
        loss = self._sess.run(self._loss,
                              feed_dict={
                                  self.data: data,
                                  self.target: target
                              })
        return loss

    def accuracy(self, data, target):
        acc = self._sess.run(self._accuracy,
                             feed_dict={
                                 self.data: data,
                                 self.target: target
                             })
        return acc

    def _predict(self, x):
        output = x
        for layer in self.layers:
            output = layer.forward(output)

        return output

    def _set_accuracy(self, y, t):
        if self._shapes[-1][1] == 1:
            correct_prediction = \
                tf.equal(tf.to_float(tf.greater(y, 0.5)), t)
        else:
            correct_prediction = \
                tf.equal(tf.argmax(y, 1), tf.argmax(t, 1))

        self._accuracy = \
            tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def _set_loss_function(self, loss):
        losses = {
            'binary_crossentropy': binary_crossentropy,
            'categorical_crossentropy': categorical_crossentropy,
            'mean_squared_error': mean_squared_error,
            'mse': mean_squared_error
        }

        if loss in losses:
            self._loss_function = losses[loss]
        else:
            self._loss_function = loss

    def _set_optimize(self, optimizer):
        optimizers = {
            'adadelta': adadelta,
            'adagrad': adagrad,
            'adam': adam,
            'momentum': momentum,
            'rmsprop': rmsprop
        }

        if optimizer in optimizers:
            self._optimize = optimizers[optimizer]
        else:
            self._optimize = optimizer
