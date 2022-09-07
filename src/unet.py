import tensorflow as tf
from tensorflow.keras.optimizers import *
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

import sys
sys.path.insert(1, '/components')

from components.encoderblock import EncoderBlock
from components.cnnblock import CNNBlock
from components.decoderblock import DecoderBlock

class Unet(tf.keras.Model):
    """
    UNet model, guide by the original paper
    https://arxiv.org/abs/1505.04597

    Implemented using best practices from tensorflow documentation
    https://www.tensorflow.org/guide/keras/custom_layers_and_models#the_model_class
    """

    def __init__(self):
        super(Unet, self).__init__()

        self.encoder1 = EncoderBlock(32)
        self.encoder2 = EncoderBlock(64)
        self.encoder3 = EncoderBlock(128)
        self.encoder4 = EncoderBlock(256)
        
        self.bottleneck = CNNBlock(512)

        self.decoder1 = DecoderBlock(256)
        self.decoder2 = DecoderBlock(128)
        self.decoder3 = DecoderBlock(64)
        self.decoder4 = DecoderBlock(32)

        self.out = Conv2D(1, 1, activation='sigmoid')
        
    def call(self, inputs):
        e1 = self.encoder1(inputs)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        b = self.bottleneck(e4)

        d1 = self.decoder1(b, self.encoder4.e)
        d2 = self.decoder2(d1, self.encoder3.e)
        d3 = self.decoder3(d2, self.encoder2.e)
        d4 = self.decoder4(d3, self.encoder1.e)

        o = self.out(d4)

        return o

    def model(self, input_shape):
        x = tf.keras.Input(shape=input_shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))