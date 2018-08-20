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

## Importable Layers, APIs

You can import low-level tftf APIs to your own TensorFlow implementations.

```python
from tftf.layers import Dense, Activation, NALU
from tftf import initializers as ini
from tftf import activations as act
from tftf import losses as loss
from tftf import optimizers as opt
from tftf.metrics import accuracy, f1

x = tf.placeholder(tf.float32, shape=[None, 784])
t = tf.placeholder(tf.float32, shape=[None, 10])

# import APIs
W = ini.glorot_normal([784, 200])  # or just write tf.Variable(...)
b = ini.zeros([200])
h = act.tanh(tf.matmul(x, W) + b)  # or just write tf.nn.tanh(...)

# import Layers
h = Dense(200)(h)
h = Activation('tanh')(h)
h = NALU(200)(h)

W = ini.glorot_normal([200, 10])
b = ini.zeros([10])
y = act.softmax(tf.matmul(h, W) + b)

cost = loss.categorical_crossentropy(y, t)
train_step = opt.sgd(0.01).minimize(cost)

# Train
#     ...

preds = y.eval(session=sess, feed_dict={x: test_X})
acc = accuracy(preds, test_y)
f = f1(preds, test_y)
print('accuracy: {:.3}'.format(acc))
print('f1: {:.3}'.format(f))
```
