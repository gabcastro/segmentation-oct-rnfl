import tensorflow as tf
from keras.optimizers import *
from keras.models import *
from keras.layers import *

class DACBlock(Layer):
    def __init__(self, out_channel, **kwargs):
        super(DACBlock, self).__init__(**kwargs)

        params_args_1 = self.get_parameters(param_kernal_size=3, param_dilation_rate=1)
        params_args_2 = self.get_parameters(param_kernal_size=3, param_dilation_rate=3)
        params_args_3 = self.get_parameters(param_kernal_size=1, param_dilation_rate=1)
        params_args_4 = self.get_parameters(param_kernal_size=3, param_dilation_rate=5)

        self.dac_branch1 = Conv2D(filters=out_channel, **params_args_1)

        self.dac_branch2_1 = Conv2D(filters=out_channel, **params_args_2)
        self.dac_branch2_2 = Conv2D(filters=out_channel, **params_args_3)
        
        self.dac_branch3_1 = Conv2D(filters=out_channel, **params_args_1)
        self.dac_branch3_2 = Conv2D(filters=out_channel, **params_args_2)
        self.dac_branch3_3 = Conv2D(filters=out_channel, **params_args_3)

        self.dac_branch4_1 = Conv2D(filters=out_channel, **params_args_1)
        self.dac_branch4_2 = Conv2D(filters=out_channel, **params_args_2)
        self.dac_branch4_3 = Conv2D(filters=out_channel, **params_args_4)
        self.dac_branch4_4 = Conv2D(filters=out_channel, **params_args_3)

        self.bn = BatchNormalization()

    def call(self, feature_map):
        dac_b1 = self.dac_branch1(feature_map)
        dac_b1 = self.bn(dac_b1)
        dac_b1 = tf.nn.relu(dac_b1)

        dac_b2 = self.dac_branch2_1(feature_map)
        dac_b2 = self.dac_branch2_2(dac_b2)
        dac_b2 = self.bn(dac_b2)
        dac_b2 = tf.nn.relu(dac_b2)

        dac_b3 = self.dac_branch3_1(feature_map)
        dac_b3 = self.dac_branch3_2(dac_b3)
        dac_b3 = self.dac_branch3_3(dac_b3)
        dac_b3 = self.bn(dac_b3)
        dac_b3 = tf.nn.relu(dac_b3)

        dac_b4 = self.dac_branch4_1(feature_map)
        dac_b4 = self.dac_branch4_2(dac_b4)
        dac_b4 = self.dac_branch4_3(dac_b4)
        dac_b4 = self.dac_branch4_4(dac_b4)
        dac_b4 = self.bn(dac_b4)
        dac_b4 = tf.nn.relu(dac_b4)

        dac_result = concatenate([dac_b1, dac_b2, dac_b3, dac_b4])

        return dac_result

    def get_config(self):
        config = super(DACBlock, self).get_config()
        config.update({
            "out_channel": self.out_channel
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def get_parameters(self, param_kernal_size, param_dilation_rate):
        return dict (
            kernel_size=param_kernal_size,
            padding='same',
            kernel_initializer='he_normal',
            dilation_rate=param_dilation_rate
        )
