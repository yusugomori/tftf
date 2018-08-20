import numpy as np
import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tftf.layers import Dense, Activation, NALU
from tftf import regularizers as reg
# from tftf.initializers import glorot_normal
from tftf import initializers as ini
from tftf import activations as act
from tftf import losses as loss
from tftf import optimizers as opt
from tftf.metrics import accuracy, f1


if __name__ == '__main__':
    np.random.seed(0)
    tf.set_random_seed(123)

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

    '''
    Build model
    '''
    n_in = 784
    n_hidden = 200
    n_out = 10

    x = tf.placeholder(tf.float32, shape=[None, n_in])
    t = tf.placeholder(tf.float32, shape=[None, n_out])

    W = ini.glorot_normal([n_in, n_hidden], name='W0')
    b = ini.zeros([n_hidden], name='b0')
    h = act.tanh(tf.matmul(x, W) + b)

    W = ini.glorot_normal([n_hidden, n_hidden], name='W1')
    b = ini.zeros([n_hidden], name='b1')
    h = act.tanh(tf.matmul(h, W) + b)

    layer = NALU(n_hidden, input_dim=n_hidden)  # import from tftf.layers
    h = layer(h)

    W = ini.glorot_normal([n_hidden, n_out], name='W_out')
    b = ini.zeros([n_out], name='b_out')
    y = act.softmax(tf.matmul(h, W) + b)

    cost = loss.categorical_crossentropy(y, t)
    train_step = opt.sgd(0.01).minimize(cost)

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
        _X, _y = shuffle(train_X, train_y)

        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size

            sess.run(train_step, feed_dict={
                x: _X[start:end],
                t: _y[start:end]
            })

        loss = cost.eval(session=sess, feed_dict={
            x: _X,
            t: _y
        })

        preds = y.eval(session=sess, feed_dict={x: _X})
        acc = accuracy(preds, _y)

        print('epoch: {}, loss: {:.3}, acc: {:.3}'.format(epoch, loss, acc))

    '''
    Test model
    '''
    preds = y.eval(session=sess, feed_dict={x: test_X})
    acc = accuracy(preds, test_y)
    f = f1(preds, test_y)
    print('accuracy: {:.3}'.format(acc))
    print('f1: {:.3}'.format(f))
