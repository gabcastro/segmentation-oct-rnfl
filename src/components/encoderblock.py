import tensorflow as tf
from tensorflow.keras.layers import *
from .cnnblock import CNNBlock

class EncoderBlock(Layer):
    def __init__(self, channel):
        super(EncoderBlock, self).__init__()

        self.cnn1 = CNNBlock(channel)
        self.cnn2 = CNNBlock(channel)

        self.pooling = MaxPooling2D()

        self.e = None

    def call(self, input_tensor, dropout = 0, training=False):
        x = self.cnn1(input_tensor, training=training)
        self.e = self.cnn2(x, training=training)
        if dropout > 0:
            self.e = Dropout(dropout)(self.e, training=training)
        return self.pooling(self.e)