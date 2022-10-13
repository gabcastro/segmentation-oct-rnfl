from keras.layers import *

from .cnnblock import CNNBlock

class DecoderBlock(Layer):
    def __init__(self, channel, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)

        self.channel = channel

        self.upsampling = UpSampling2D()
        self.cnn1 = CNNBlock(self.channel)
        self.cnn2 = CNNBlock(self.channel)

    def call(self, input_tensor, concat_tensor):
        x = self.upsampling(input_tensor)
        x = concatenate([concat_tensor, x], axis=3)
        x = self.cnn1(x)
        return self.cnn2(x)

    def get_config(self):
        config = super(DecoderBlock, self).get_config()
        config.update({
            "channel": self.channel,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)