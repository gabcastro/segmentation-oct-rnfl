import tensorflow as tf
from tensorflow.keras.layers import *

from .cnnblock import CNNBlock

class DecoderBlock(Layer):
    def __init__(self, channel):
        super(DecoderBlock, self).__init__()

        self.upsampling = UpSampling2D()
        self.cnn1 = CNNBlock(channel)
        self.cnn2 = CNNBlock(channel)

    def call(self, input_tensor, concat_tensor, training=False):
        x = self.upsampling(input_tensor)
        x = concatenate([x, concat_tensor])
        x = self.cnn1(x, training=training)
        return self.cnn2(x, training=training)