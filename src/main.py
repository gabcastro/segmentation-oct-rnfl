import cv2
import numpy as np

from numpy import dtype
from dataset import Dataset
from unet import Unet
from compile import Compile

import tensorflow as tf

def main():

    ds = Dataset('../data/v2/L1/train')
    ds_train = ds.create_dataset()

    shape=(512, 512, 1)
    input = tf.keras.Input(shape=shape)

    unet = Unet()
    unet(input)

    unet.model(input_shape=shape).summary()

    compile_methods = Compile()

    unet.compile(
        optimizer=compile_methods.optimizer,
        loss=compile_methods.loss,
        metrics=compile_methods.all_metrics
    )

    history = unet.fit(
        x=ds_train,
        steps_per_epoch=len(ds_train),
        epochs=1,
        #validation_data=ds_val_zip,
        #validation_steps=len(ds_val_zip),
        verbose=1
    )

if __name__ == "__main__":
    main()