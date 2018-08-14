import os
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from ..layers import Dense
from .losses import *
from .optimizers import *


class Model(object):
    def __init__(self, reset_graph=True):
        if reset_graph:
            tf.reset_default_graph()
        self._layers = []
        self._shapes = []
        self._sess = None
        self._restored = False

    def __del__(self):
        if self._sess is not None:
            self._sess.close()

    @property
    def layers(self):
        return self._layers

    def add(self, layer):
        input_dim = layer.input_dim
        output_dim = layer.output_dim

        if input_dim is None:
            if len(self.layers) == 0:
                raise AttributeError('input_dim must be specified '
                                     'on first layer.')
            else:
                layer.input_dim = self._shapes[-1][1]

        if output_dim is None:
            layer.initialize_output_dim()

        self._shapes.append(layer.shape)
        self._layers.append(layer)

    def compile(self, loss='mse', optimizer='rmsprop'):
        if not self._restored:
            for layer in self._layers:
                layer.compile()

        self._set_loss_function(loss)
        self._set_optimize(optimizer)

        input_shape = [None] + list(self.layers[0].input_shape)
        output_shape = [None] + list(self.layers[-1].output_shape)

        x = self.data = tf.placeholder(tf.float32, shape=input_shape)
        t = self.target = tf.placeholder(tf.float32, shape=output_shape)
        training = self.training = \
            tf.placeholder_with_default(False, ())

        y = self._y = self._predict(x, training=training)
        self._loss = self._loss_function(y, t)
        self._set_accuracy(y, t)
        self._train_step = self._optimize().minimize(self._loss)

        if not self._restored:
            self._sess = tf.Session()
            self._init = tf.global_variables_initializer()
            self._sess.run(self._init)

    def describe(self):
        layers = self.layers
        digits = int(np.log10(len(layers))) + 1
        for i, layer in enumerate(layers):
            print('#{}: {}'.format(str(i).zfill(digits), layer))

    def describe_params(self):
        layers = self.layers
        digits = int(np.log10(len(layers))) + 1
        for i, layer in enumerate(layers):
            _params = layer.params
            print('-' * 48)
            print('#{}: {}'.format(str(i).zfill(digits), layer))
            print('-' * 48)
            if len(_params) == 0:
                print('No params')
            else:
                for j, param in enumerate(_params):
                    print('{}: {}'.format(param.name,
                                          param.get_shape()))
            if i == len(layers) - 1:
                print('-' * 48)

    def eval(self, elem, feed_dict):
        return self._sess.run(elem, feed_dict=feed_dict)

    # TODO: validation data
    def fit(self, data, target,
            epochs=10, batch_size=100,
            verbose=1):
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

                self.eval(self._train_step,
                          feed_dict={
                              self.data: _data[_start:_end],
                              self.target: _target[_start:_end],
                              self.training: True
                          })
            _loss = self.loss(_data, _target)
            _acc = self.accuracy(_data, _target)
            if verbose:
                print('Epoch: {}, loss: {:.3}, acc: {:.3}'
                      .format((epoch + 1), _loss, _acc))

    def predict(self, data):
        ret = self.eval(self._y,
                        feed_dict={
                            self.data: data
                        })
        return ret

    def loss(self, data, target):
        loss = self.eval(self._loss,
                         feed_dict={
                            self.data: data,
                            self.target: target
                         })
        return loss

    def accuracy(self, data, target):
        acc = self.eval(self._accuracy,
                        feed_dict={
                            self.data: data,
                            self.target: target
                        })
        return acc

    def restore(self, model_path):
        if self._sess is not None:
            raise AttributeError('Session alrady initialized. '
                                 'Model variables must be restored '
                                 'before compile.')
        for layer in self._layers:
            layer.compile()

        self._sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(self._sess, model_path)
        self._restored = True

    def save(self, out_path, verbose=1):
        out_dir = out_path.split('/')[:-1]
        if len(out_dir) > 0:
            os.makedirs(os.path.join(*out_dir), exist_ok=True)
        saver = tf.train.Saver()
        saver.save(self._sess, out_path)

        if verbose:
            print('Model saved to: \'{}\''.format(out_path))

    def _predict(self, x, **kwargs):
        output = x
        for layer in self.layers:
            output = layer.forward(output, **kwargs)

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
