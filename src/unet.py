import tensorflow as tf
from tensorflow.python.keras.optimizers import *
from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *

class Unet(tf.keras.Model):
    """
    UNet model, guinde by the original paper
    https://arxiv.org/abs/1505.04597

    Implemented using best practices from tensorflow documentation
    https://www.tensorflow.org/guide/keras/custom_layers_and_models#the_model_class
    """

    def __init__(self):
        super(Unet, self).__init__()

        params = dict(
            kernel_size=(3, 3),
            activation='relu',
            padding='same',
            kernel_initializer='he_normal',
            dilation_rate=1
        )
        
        self.conv1a_e = Conv2D(name='conv1a_e', filters=64, **params)
        self.conv1b_e = Conv2D(name='conv1b_e', filters=64, **params)
        self.pool1 = MaxPooling2D(name='pool1', pool_size=(2, 2))

        self.conv2a_e = Conv2D(name='conv2a_e', filters=128, **params)
        self.conv2b_e = Conv2D(name='conv2b_e', filters=128, **params)
        self.pool2 = MaxPooling2D(name='pool2', pool_size=(2, 2))

        self.conv3a_e = Conv2D(name='conv3a_e', filters=256, **params)
        self.conv3b_e = Conv2D(name='conv3b_e', filters=256, **params)
        self.pool3 = MaxPooling2D(name='pool3', pool_size=(2, 2))

        self.conv4a_e = Conv2D(name='conv4a_e', filters=512, **params)
        self.conv4b_e = Conv2D(name='conv4b_e', filters=512, **params)
        self.pool4 = MaxPooling2D(name='pool4', pool_size=(2, 2))

        self.conv5a = Conv2D(name='conv5a_bottleneck', filters=1024, **params)
        self.conv5b = Conv2D(name='conv5b_bottleneck', filters=1024, **params)

        self.upsampling1 = UpSampling2D(name='upsampling1', size=(2, 2))
        self.upsampling2 = UpSampling2D(name='upsampling2', size=(2, 2))
        self.upsampling3 = UpSampling2D(name='upsampling3', size=(2, 2))
        self.upsampling4 = UpSampling2D(name='upsampling4', size=(2, 2))

        self.conv1a_d = Conv2D(name='conv1a_d', filters=512, **params)
        self.conv1b_d = Conv2D(name='conv1b_d', filters=512, **params)

        self.conv2a_d = Conv2D(name='conv2a_d', filters=256, **params)
        self.conv2b_d = Conv2D(name='conv2b_d', filters=256, **params)

        self.conv3a_d = Conv2D(name='conv3a_d', filters=128, **params)
        self.conv3b_d = Conv2D(name='conv3b_d', filters=128, **params)

        self.conv4a_d = Conv2D(name='conv4a_d', filters=64, **params)
        self.conv4b_d = Conv2D(name='conv4b_d', filters=64, **params)

        self.out = Conv2D(name='output', filters=1, kernel_size=(1, 1), activation='sigmoid')
        
    def call(self, inputs):
        e1 = self.conv1a_e(inputs)
        e1 = self.conv1b_e(e1) #64
        p1 = self.pool1(e1)

        e2 = self.conv2a_e(p1)
        e2 = self.conv2b_e(e2) #128
        p2 = self.pool2(e2)

        e3 = self.conv3a_e(p2)
        e3 = self.conv3b_e(e3) #256
        p3 = self.pool3(e3)

        e4 = self.conv4a_e(p3)
        e4 = self.conv4b_e(e4) #512
        p4 = self.pool4(e4)

        e5 = self.conv5a(p4)
        e5 = self.conv5b(e5) #1024

        u1 = self.upsampling1(e5)
        c1 = concatenate([u1, e4]) #512
        d1 = self.conv1a_d(c1)
        d1 = self.conv1b_d(d1)

        u2 = self.upsampling2(d1)
        c2 = concatenate([u2, e3]) #256
        d2 = self.conv2a_d(c2)
        d2 = self.conv2b_d(d2)

        u3 = self.upsampling3(d2)
        c3 = concatenate([u3, e2]) #128
        d3 = self.conv3a_d(c3)
        d3 = self.conv3b_d(d3)

        u4 = self.upsampling4(d3)
        c4 = concatenate([u4, e1]) #64
        d4 = self.conv4a_d(c4)
        d4 = self.conv4b_d(d4)

        o = self.out(d4)

        return o