import tensorflow as tf
from . import Module
from .. import Activation
from .. import BatchNormalization
from .. import Conv2D
from .. import Dense
from .. import GlobalAveragePooling2D
from .. import MaxPooling2D


class ResNet(Module):
    '''
    # Example

        ```
        x = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
        t = tf.placeholder(tf.float32, shape=[None, 10])

        resnet = ResNet()
        h = resnet.v1(x)
        h = Activation('relu')(h)
        h = Dense(10)(h)
        y = Activation('softmax')(h)

        cost = categorical_crossentropy(y, t)
        train_step = sgd(0.01).minimize(cost)
        ```
    '''
    def __init__(self):
        pass

    def v1(self, x, n_out=1000):
        '''
        ResNet-34

        # Arguments
            x: placeholder
        '''
        layers = [
            Conv2D(kernel_size=(7, 7, 64)),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(pool_size=(3, 3),
                         strides=(2, 2),
                         padding='same')
        ]
        for layer in layers:
            x = layer(x)
        x = self._add_base_block(x, channel_out=64)
        x = self._add_base_block(x, channel_out=128)
        x = self._add_base_block(x, channel_out=256)
        x = self._add_base_block(x, channel_out=512)
        x = GlobalAveragePooling2D()(x)
        x = Dense(n_out)(x)

        return x

    def _add_base_block(self, x, channel_out=64):
        x = Conv2D(kernel_size=(1, 1, channel_out),
                   strides=(2, 2))(x)
        x = self._base_block(x, channel_out=channel_out)
        return x

    def _base_block(self, x, channel_out=64):
        '''
        # Arguments
            x: placeholder
        '''
        layers = [
            Conv2D(kernel_size=(3, 3, channel_out)),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(kernel_size=(3, 3, channel_out)),
            BatchNormalization()
        ]
        for layer in layers:
            h = layer(x)
        shortcut = self._shortcut(x, output_shape=h.get_shape())

        return Activation('relu')(h + shortcut)

    def _bottleneck(self, x, channel_out=256):
        '''
        # Arguments
            x: placeholder
        '''
        channel = channel_out // 4
        layers = [
            Conv2D(kernel_size=(1, 1, channel)),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(kernel_size=(3, 3, channel)),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(kernel_size=(1, 1, channel)),
            BatchNormalization()
        ]
        for layer in layers:
            h = layer(x)
        shortcut = self._shortcut(x, output_shape=h.get_shape())

        return Activation('relu')(h + shortcut)

    def _projection(self, x, channel_out):
        layer = Conv2D(kernel_size=(1, 1, channel_out))
        return layer(x)

    def _shortcut(self, x, output_shape):
        input_shape = x.get_shape()
        channel_in = input_shape[-1]
        channel_out = output_shape[-1]

        if channel_in != channel_out:
            return self._projection(x, channel_out)
        else:
            return x
