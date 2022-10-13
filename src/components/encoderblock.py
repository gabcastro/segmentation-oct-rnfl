from keras.layers import *
from .cnnblock import CNNBlock

class EncoderBlock(Layer):
    def __init__(self, channel, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)

        self.channel = channel

        self.cnn1 = CNNBlock(self.channel)
        self.cnn2 = CNNBlock(self.channel)

        self.pooling = MaxPooling2D()

    def call(self, input_tensor, dropout = 0):
        x = self.cnn1(input_tensor)
        x = self.cnn2(x)
        if dropout > 0:
            x = Dropout(dropout)(x)
        y = self.pooling(x)
        return x, y

    def get_config(self):
        config = super(EncoderBlock, self).get_config()
        config.update({
            "channel": self.channel,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)