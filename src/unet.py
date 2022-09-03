from platform import release
import tensorflow as tf
from tensorflow import Tensor
from tensorflow.python.keras import backend as keras
from tensorflow.python.keras.optimizers import *
from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.callbacks import ModelCheckpoint, LearningRateScheduler

class Unet:
    """UNet model, guinde by the original paper"""

    def __init__(self, shape):
        self.input = tf.keras.Input(shape=shape)

    def relu_bn(self, inputs: Tensor) -> Tensor:
        relu = ReLU()(inputs)
        bn = BatchNormalization()(relu)
        return bn

    def max_pooling(self,
                    x: Tensor,
                    size: tuple = (2, 2),
                    strides = None) -> Tensor:
        y = MaxPooling2D(pool_size=size, strides=strides)(x)

        return y

    def conv2D_block(self, 
                     x: Tensor, 
                     filter_size: int, 
                     kernel_size: int = 3, 
                     dilation_rate: int = 1,
                     reluBn = True) -> Tensor:
        y = Conv2D(filters=filter_size,
                    kernel_size=kernel_size,
                    dilation_rate=dilation_rate,
                    padding="same",
                    kernel_initializer="he_normal")(x)
        if reluBn:
            y = self.relu_bn(y)

        return y

    def upsampling_concat_block(self,
                                upConvBlock: Tensor, 
                                concatBlock: Tensor,
                                filter_size: int, 
                                kernel_size: int = 3, 
                                dilation_rate: int = 1,
                                isConcat: bool = True,
                                interpolation: str = "nearest",
                                upSamplingSize: tuple = (2, 2)) -> Tensor:
        y = UpSampling2D(size=upSamplingSize, interpolation=interpolation)(upConvBlock)
        
        if isConcat:
            y = concatenate([concatBlock, y], axis=3)
        
        return y

    