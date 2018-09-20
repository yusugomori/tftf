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
    test_size = 100     # up to 500
    train_X, train_y = train_X[:train_size], train_y[:train_size]
    test_X, test_y = test_X[:test_size], test_y[:test_size]

    '''
    Build model
    '''
    pad_value = 0
    x = tf.placeholder(tf.int32, [None, None], name='x')
    t = tf.placeholder(tf.int32, [None, None], name='t')

    transformer = Transformer(num_X, num_y)
    preds = transformer.v1(x, t)

    cost = transformer.loss()
    optimizer, lr = transformer.optimizer(cost)
    train = transformer.is_training

    '''
    Train model
    '''
    epochs = 30
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
    maxlen = 100
    y = tf.placeholder(tf.int32, [None, None])
    step = tf.constant(0)
    flg = tf.cast(tf.zeros_like(y[:, 0]), dtype=tf.bool)

    mask_src = transformer._pad_mask(x)
    memory = transformer.encode(x, mask=mask_src)

    def cond(y, step, f):
        n_flg = tf.reduce_sum(tf.cast(f, tf.int32))
        next = \
            tf.not_equal(n_flg,
                         tf.reduce_sum(tf.ones_like(flg,
                                                    dtype=tf.int32)))
        return tf.logical_and(step+1 < maxlen, next)

    def body(y, step, f):
        mask_tgt = transformer._pad_subsequent_mask(y)
        h = transformer.decode(y, memory,
                               mask_src=mask_src, mask_tgt=mask_tgt,
                               recurrent=True)
        output = transformer.generate(h[:, -1], recurrent=False)
        output = tf.cast(tf.argmax(output, axis=1), tf.int32)
        y = tf.concat([y, output[:, np.newaxis]], axis=1)
        f = tf.logical_or(f, tf.equal(output, end_char))

        return [y, step+1, f]

    generator = tf.while_loop(cond,
                              body,
                              loop_vars=[y, step, flg],
                              shape_invariants=[
                                tf.TensorShape([None, None]),
                                step.get_shape(),
                                tf.TensorShape([None])])

    test_X_ = pad_sequences(test_X, value=pad_value)
    y_ = start_char * np.ones_like(test_X, dtype='int32')[:, np.newaxis]
    preds, _, _ = sess.run(generator, feed_dict={
        x: test_X_,
        y: y_
    })

    for n in range(len(test_X)):
        data = test_X[n][1:-1]
        target = test_y[n][1:-1]
        pred = list(preds[n])[1:]
        pred.append(end_char)

        print('-' * 20)
        print('Original sentence:',
              ' '.join([i2w_X[i] for i in data]))
        print('True sentence:',
              ' '.join([i2w_y[i] for i in target]))
        print('Generated sentence:',
              ' '.join([i2w_y[i] for i in pred[:pred.index(end_char)]]))
