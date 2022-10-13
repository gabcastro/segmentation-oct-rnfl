import tensorflow as tf
from keras.optimizers import *
from keras.models import *
from keras.layers import *

class CNNBlock(Layer):
    def __init__(self, out_channel, params_args = None, kernel_size=3, **kwargs):
        super(CNNBlock, self).__init__(**kwargs)

        self.out_channel = out_channel
        self.params_args = {}
        self.kernel_size = kernel_size

        if params_args is None:
            params = dict (
                padding='same',
                kernel_initializer='he_normal',
                dilation_rate=1
            )
        else:
            params = params_args

        self.conv = Conv2D(filters=self.out_channel, kernel_size=self.kernel_size, **params)
        self.bn = BatchNormalization()

    def call(self, input_tensor):
        x = self.conv(input_tensor)
        x = self.bn(x)
        x = tf.nn.relu(x)
        return x

    def get_config(self):
        config = super(CNNBlock, self).get_config()
        config.update({
            "out_channel": self.out_channel,
            "params_args": self.params_args,
            "kernel_size": self.kernel_size,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
