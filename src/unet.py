import tensorflow as tf
from keras.optimizers import *
from keras.models import *
from keras.layers import *

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
        e1x, e1y = self.encoder1(inputs)
        e2x, e2y = self.encoder2(e1y, 0.2)
        e3x, e3y = self.encoder3(e2y, 0.3)
        e4x, e4y = self.encoder4(e3y, 0.7)

        b = self.bottleneck(e4y)

        d1 = self.decoder1(b, e4x)
        d2 = self.decoder2(d1, e3x)
        d3 = self.decoder3(d2, e2x)
        d4 = self.decoder4(d3, e1x)

        o = self.out(d4)

        return o

    def model(self, input_shape):
        x = tf.keras.Input(shape=input_shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

    def modelcheckpoint(self, filepath):
        return tf.keras.callbacks.ModelCheckpoint(
            filepath=filepath,
            verbose=1,
            save_best_only=True,
            monitor="dice_coef"
        )