import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tftf.layers import Dense, Activation
from tftf.layers import LSTM, Embedding, TimeDistributedDense, Attention
from tftf.preprocessing.sequence import pad_sequences, sort
from tftf.datasets import load_small_parallel_enja
from tftf.layers.modules import Transformer

if __name__ == '__main__':
    np.random.seed(0)
    tf.set_random_seed(123)

    '''
    Load data
    '''
    start_char = 1
    end_char = 2
    (train_X, train_y), (test_X, test_y), (num_X, num_y), \
        (w2i_X, w2i_y), (i2w_X, i2w_y) = load_small_parallel_enja(to_ja=True)

    train_X, train_y = sort(train_X, train_y)
    test_X, test_y = sort(test_X, test_y)

    train_size = 50000  # up to 50000
    test_size = 500     # up to 500
    train_X, train_y = train_X[:train_size], train_y[:train_size]
    test_X, test_y = test_X[:test_size], test_y[:test_size]

    '''
    Build model
    '''
    pad_value = 0
    x = tf.placeholder(tf.int32, [None, None], name='x')
    t = tf.placeholder(tf.int32, [None, None], name='t')

    transformer = Transformer(num_X, num_y, N=1)
    preds = transformer.v1(x, t)

    cost = transformer.loss()
    optimizer, lr = transformer.optimizer(cost)
    train = transformer.is_training

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

            _, _cost = sess.run([optimizer, cost], feed_dict={
                x: _train_X,
                t: _train_y,
                lr: transformer.lrate(epoch),
                train: True
            })
            loss += _cost

        loss /= n_batches

        _test_X = pad_sequences(test_X, value=pad_value)
        _test_y = pad_sequences(test_y, value=pad_value)

        val_loss = cost.eval(session=sess, feed_dict={
            x: _test_X,
            t: _test_y
        })

        print('epoch: {}, '
              'loss: {:.3}, '
              'val_loss: {:.3}'.format(epoch+1, loss, val_loss))

    '''
    Generate sentences
    '''
    preds = transformer.greedy_decode(maxlen=100)
    test_X_ = pad_sequences(test_X, value=pad_value)
    test_y_ = pad_sequences(test_y, value=pad_value)

    preds = sess.run(preds, feed_dict={
        x: test_X_,
        t: test_y_
    })

    for n in range(len(test_X)):
        data = test_X[n][1:-1]
        target = test_y[n][1:-1]
        pred = list(preds[n])
        pred.append(end_char)
        print('-' * 20)
        print('Original sentence:',
              ' '.join([i2w_X[i] for i in data]))
        print('True sentence:',
              ' '.join([i2w_y[i] for i in target]))
        print('Generated sentence:',
              ' '.join([i2w_y[i] for i in pred[:pred.index(end_char)]]))
