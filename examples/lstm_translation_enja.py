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
    end_char = 2
    (train_X, train_y), (test_X, test_y), (num_X, num_y), \
        (w2i_X, w2i_y), (i2w_X, i2w_y) = load_small_parallel_enja(to_ja=True)

    train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y)

    train_X, train_y = sort(train_X, train_y)
    valid_X, valid_y = sort(valid_X, valid_y)
    test_X, test_y = sort(test_X, test_y)

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
        LSTM(256, return_sequence=True, return_cell=True)
    ]

    h = x
    for layer in encoder:
        h = layer(h, mask=mask_enc)
    encoder_output, encoder_cell = h

    decoder = [
        [
            Embedding(256, input_dim=num_y),
            LSTM(256, return_sequence=True, return_cell=True,
                 initial_state=encoder_output[:, -1, :],
                 cell_state=encoder_cell[:, -1, :])
        ],
        [
            TimeDistributedDense(num_y),
            Activation('softmax')
        ]
    ]

    h = t[:, :-1]
    for layer in decoder[0]:
        h = layer(h)
    decoder_output, _ = h

    output = decoder_output
    for layer in decoder[1]:
        output = layer(output)

    cost = \
        loss.categorical_crossentropy(output,
                                      target
                                      * tf.transpose(mask_dec[:, np.newaxis],
                                                     perm=[0, 2, 1]))
    train_step = opt.adam().minimize(cost)

    '''
    Train model
    '''
    epochs = 1
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

            _, _cost = sess.run([train_step, cost], feed_dict={
                x: _train_X,
                t: _train_y
            })
            loss += _cost

        loss /= n_batches

        _valid_X = pad_sequences(valid_X, value=pad_value)
        _valid_y = pad_sequences(valid_y, value=pad_value)

        val_loss = cost.eval(session=sess, feed_dict={
            x: _valid_X,
            t: _valid_y
        })

        print('epoch: {}, '
              'loss: {:.3}, '
              'val_loss: {:.3}'.format(epoch+1, loss, val_loss))

    '''
    Generate sentences
    '''
    initial = {
        'y': tf.placeholder(tf.int32, [None, None]),
        'state': tf.placeholder(tf.float32, [None, None]),
        'cell_state': tf.placeholder(tf.float32, [None, None]),
        'step': tf.constant(0)
    }
    initial['flg'] = tf.cast(tf.zeros_like(initial['y'][:, 0]), dtype=tf.bool)
    maxlen = 100

    def cond(y, state, cell_state, step, flg):
        n_flg = tf.reduce_sum(tf.cast(flg, tf.int32))
        next = \
            tf.not_equal(n_flg,
                         tf.reduce_sum(tf.ones_like(initial['flg'],
                                                    dtype=tf.int32)))
        return tf.logical_and(step+1 < maxlen, next)

    def body(y, state, cell_state, step, flg):
        h = y[:, -1]
        for layer in decoder[0]:
            h = layer(h,
                      recurrent=False,
                      initial_state=state,
                      cell_state=cell_state)
        decoder_output, decoder_cell = h

        output = decoder_output
        for layer in decoder[1]:
            output = layer(output, recurrent=False)
        output = tf.cast(tf.argmax(output, axis=1), tf.int32)
        y = tf.concat([y, output[:, np.newaxis]], axis=1)
        flg = tf.logical_or(flg, tf.equal(output, end_char))

        return [y,
                decoder_output,
                decoder_cell,
                step+1,
                flg]
    generator = \
        tf.while_loop(cond,
                      body,
                      loop_vars=[initial['y'],
                                 initial['state'],
                                 initial['cell_state'],
                                 initial['step'],
                                 initial['flg']],
                      shape_invariants=[tf.TensorShape([None, None]),
                                        tf.TensorShape([None, None]),
                                        tf.TensorShape([None, None]),
                                        initial['step'].get_shape(),
                                        tf.TensorShape([None])])

    test_X_ = pad_sequences(test_X, value=pad_value)
    init_y = np.zeros_like(test_X, dtype='int32')[:, np.newaxis]
    state, cell_state = \
        sess.run([encoder_output, encoder_cell], feed_dict={
            x: test_X_
        })
    init_state = state[:, -1, :]
    init_cell_state = cell_state[:, -1, :]

    preds, _, _, _, _ = sess.run(generator, feed_dict={
        initial['y']: init_y,
        initial['state']: init_state,
        initial['cell_state']: init_cell_state
    })

    n = 0
    data = test_X[n][1:-1]
    target = test_y[n][1:-1]
    preds = list(preds[n])[1:]
    preds.append(end_char)

    print('Original sentence:',
          ' '.join([i2w_X[i] for i in data]))
    print('True sentence:',
          ' '.join([i2w_y[i] for i in target]))
    print('Generated sentence:',
          ' '.join([i2w_y[i] for i in preds[:preds.index(end_char)]]))
