from dataset import Dataset
from unet import Unet
from compile import Compile

import tensorflow as tf
from keras.models import load_model

import sys
sys.path.insert(1, '/components')

from components.encoderblock import EncoderBlock
from components.cnnblock import CNNBlock
from components.decoderblock import DecoderBlock

def main():

    ds = Dataset('../data/v2/L1/train')
    ds_train = ds.create_dataset(augment=True)

    ds = Dataset('../data/v2/L1/validation')
    ds_val = ds.create_dataset()

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
        validation_data=ds_val,
        validation_steps=len(ds_val),
        verbose=1
    )

    # TODO: necessário ver ainda se será um problema os warnings ao fazer o load_model 
    
    unet.save("./tmp/model/", save_format="tf")

    new_model = load_model(
        './tmp/model/', 
        compile=False,
        custom_objects={
            "CNNBlock": CNNBlock,
            "EncoderBlock": EncoderBlock,
            "DecoderBlock": DecoderBlock,
            "Unet": Unet, 
        }
    )

if __name__ == "__main__":
    main()