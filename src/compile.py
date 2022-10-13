import tensorflow as tf
from keras.layers import Flatten

smooth = 1e-15
lr = 1e-4

def dice_coef(y_true, y_pred):
    y_true = Flatten()(y_true)
    y_pred = Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)

    numerator = 2. * intersection + smooth
    denominator = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth

    result = numerator / denominator
    
    return result

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)