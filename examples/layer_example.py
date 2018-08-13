import numpy as np
import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tftf.layers import Dense, Activation, NALU


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

    W = tf.Variable(tf.truncated_normal([n_in, n_hidden], stddev=0.1))
    b = tf.Variable(tf.zeros([n_hidden]))
    h = tf.nn.tanh(tf.matmul(x, W) + b)

    # add layers with tftf.layers
    layer = Dense(n_hidden, input_dim=n_hidden)
    layer.compile()
    h = layer.forward(h)

    layer = Activation('tanh')
    layer.compile()
    h = layer.forward(h)

    layer = NALU(n_hidden, input_dim=n_hidden)
    layer.compile()
    h = layer.forward(h)

    W = tf.Variable(tf.truncated_normal([n_hidden, n_out], stddev=0.1))
    b = tf.Variable(tf.zeros([n_out]))
    y = tf.nn.softmax(tf.matmul(h, W) + b)

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(t * tf.log(y), axis=1))
    train_step = \
        tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(t, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

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

        loss = cross_entropy.eval(session=sess, feed_dict={
            x: _X,
            t: _y
        })
        acc = accuracy.eval(session=sess, feed_dict={
            x: _X,
            t: _y
        })
        print('epoch:', epoch, ' loss:', loss, ' accuracy:', acc)

    '''
    Test model
    '''
    accuracy_rate = accuracy.eval(session=sess, feed_dict={
        x: test_X,
        t: test_y
    })
    print('accuracy: ', accuracy_rate)
