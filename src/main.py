from dataset import Dataset
from unet import Unet
from compile import dice_coef, dice_loss
from evaluate import Evaluate

import tensorflow as tf
from keras.models import load_model
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import Adam

import sys
sys.path.insert(1, '/components')

from components.encoderblock import EncoderBlock
from components.cnnblock import CNNBlock
from components.decoderblock import DecoderBlock

import cv2
import numpy as np

def main():

    running = False

    if (running):
        lr=1e-4

        ds = Dataset('../data/v2/L1/train', aug_read=True)
        ds_train = ds.create_dataset()

        ds = Dataset('../data/v2/L1/validation')
        ds_val = ds.create_dataset()

        shape=(512, 512, 1)
        input = tf.keras.Input(shape=shape)

        unet = Unet()
        unet(input)

        unet.model(input_shape=shape).summary()

        unet.compile(
            optimizer=Adam(lr),
            loss=dice_loss,
            metrics=[dice_coef]
        )

        callback = [
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
        ]

        history = unet.fit(
            x=ds_train,
            steps_per_epoch=len(ds_train),
            epochs=1,
            validation_data=ds_val,
            validation_steps=len(ds_val),
            callbacks=callback,
            verbose=1
        )
    
        unet.save("../tmp/model/", save_format="tf")
    else:
        unet = load_model(
            '../tmp/model/', 
            compile=False,
            custom_objects={
                "CNNBlock": CNNBlock,
                "EncoderBlock": EncoderBlock,
                "DecoderBlock": DecoderBlock,
                "Unet": Unet, 
            }
        )

    eval = Evaluate('../data/v2/L1/test', '../data/v2/L1/predicted')
    eval.eval(unet)

if __name__ == "__main__":
    main()