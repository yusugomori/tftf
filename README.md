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
model.add(Dense(784, 500))
model.add(Activation('sigmoid'))
model.add(Dense(500, 10))
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
