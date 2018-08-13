# TFTF: TensorFlow TransFormer🍔

TensorFlow for everybody.

## Quick glance

```python
from tftf.layers import Layer, Dense, Activation
from tftf.models import Model

'''
Build model
'''
model = Model()
model.add(Dense(500, input_dim=784))
model.add(Activation('sigmoid'))
model.add(Dense(10))
model.add(Activation('softmax'))
model.compile()

model.describe()

'''
Train model
'''
model.fit(train_X, train_y)

'''
Test model
'''
print(model.accuracy(test_X, test_y))
```

See [examples](https://github.com/yusugomori/tftf/tree/master/examples) for other implementations.

## Installation

- **Install TFTF from PyPI (recommended):**

```sh
pip install tensorflow
pip install tftf
```

- **Alternatively: install TFTF from the GitHub source:**

First, clone TFTF using `git`:

```sh
git clone https://github.com/yusugomori/tftf.git
```

 Then, `cd` to the TFTF folder and run the install command:
```sh
cd tftf
sudo python setup.py install
```

## Importable Layer

You can just use `tftf.layers` to your own TensorFlow implementations.

```python
from tftf.layers import Dense, Activation, NALU

x = tf.placeholder(tf.float32, shape=[None, 784])
t = tf.placeholder(tf.float32, shape=[None, 10])

W = tf.Variable(tf.truncated_normal([784, 200], stddev=0.1))
b = tf.Variable(tf.zeros([200]))
h = tf.nn.tanh(tf.matmul(x, W) + b)

# add tftf.layers into pure TF implementations
layer = Dense(200, input_dim=200)
layer.compile()
h = layer.forward(h)

layer = Activation('tanh')
layer.compile()
h = layer.forward(h)

layer = NALU(200, input_dim=200)
layer.compile()
h = layer.forward(h)

W = tf.Variable(tf.truncated_normal([200, 10], stddev=0.1))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(h, W) + b)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(t * tf.log(y), axis=1))
train_step = \
    tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(t, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
```
