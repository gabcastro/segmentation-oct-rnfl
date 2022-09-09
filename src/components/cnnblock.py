import tensorflow as tf
from tensorflow.keras.optimizers import *
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

class CNNBlock(Layer):
    def __init__(self, out_channel, params_args = None, kernel_size=3):
        super(CNNBlock, self).__init__()

        params = {}

        if params_args is None:
            params = dict (
                padding='same',
                kernel_initializer='he_normal',
                dilation_rate=1
            )
        else:
            params = params_args

        self.conv = Conv2D(filters=out_channel, kernel_size=kernel_size, **params)
        self.bn = BatchNormalization()

    def call(self, input_tensor):
        x = self.conv(input_tensor)
        x = self.bn(x)
        x = tf.nn.relu(x)
        return x
