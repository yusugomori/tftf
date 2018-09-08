import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tftf.layers import Dense, Activation
from tftf.layers import LSTM, Embedding, TimeDistributedDense
from tftf.preprocessing.sequence import pad_sequences, sort
from tftf.datasets import load_small_parallel_enja
from tftf import losses as loss
from tftf import optimizers as opt

if __name__ == '__main__':
    np.random.seed(0)
    tf.set_random_seed(123)

    '''
    Load data
    '''
    (train_X, train_y), (test_X, test_y), (num_X, num_y) = \
        load_small_parallel_enja(to_ja=True)

    train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y)

    train_X, train_y = sort(train_X, train_y)
    valid_X, valid_y = sort(valid_X, valid_y)
    test_X, test_y = sort(test_X, test_y)

    train_size = 10000  # up to 37500
    valid_size = 5000   # up to 12500
    train_X, train_y = train_X[:train_size], train_y[:train_size]
    valid_X, valid_y = valid_X[:valid_size], valid_y[:valid_size]

    '''
    Build model
    '''
    pad_value = 0
    x = tf.placeholder(tf.int32, [None, None], name='x')
    t = tf.placeholder(tf.int32, [None, None], name='t')
    target = tf.one_hot(t[:, 1:], depth=num_y, dtype=tf.float32)
    mask_enc = tf.cast(tf.not_equal(x, pad_value), tf.float32)
    mask_dec = tf.cast(tf.not_equal(t[:, 1:], pad_value), tf.float32)

    encoder = [
        Embedding(256, input_dim=num_X),
        LSTM(256, return_sequence=True)
    ]

    h = x
    for layer in encoder:
        h = layer(h, mask=mask_enc)

    decoder = [
        Embedding(256, input_dim=num_y),
        LSTM(256, return_sequence=True,
             initial_state=h[:, -1, :],
             cell_state=encoder[-1].cell_state[:, -1, :])
    ]

    h = t[:, :-1]
    for layer in decoder:
        h = layer(h)

    h = TimeDistributedDense(num_y)(h)
    y = Activation('softmax')(h)

    cost = \
        loss.categorical_crossentropy(y,
                                      target
                                      * tf.transpose(mask_dec[:, np.newaxis],
                                                     perm=[0, 2, 1]))
    train_step = opt.adam().minimize(cost)

    '''
    Train model
    '''
    epochs = 10
    batch_size = 100

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    n_batches = len(train_X) // batch_size

    for epoch in range(epochs):
        loss = 0.
        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size

            _train_X = pad_sequences(train_X[start:end],
                                     value=pad_value)
            _train_y = pad_sequences(train_y[start:end],
                                     value=pad_value)

            _, _loss = sess.run([train_step, cost], feed_dict={
                x: _train_X,
                t: _train_y
            })
            loss += _loss

        loss /= n_batches

        _valid_X = pad_sequences(valid_X, value=pad_value)
        _valid_y = pad_sequences(valid_y, value=pad_value)

        val_loss = cost.eval(session=sess, feed_dict={
            x: _valid_X,
            t: _valid_y
        })

        print('epoch: {}, '
              'loss: {:.3}, '
              'val_loss: {:.3}'.format(epoch + 1, loss, val_loss))
