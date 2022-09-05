from unet import Unet
from compile import Compile
from datagenerator import DataGenerator
import tensorflow as tf

def main():
    datagen = DataGenerator()
    train_gen = datagen.generetor(
        directory='', 
        folders=['originals', 'masks'], 
        dirToSave='')

    input = tf.keras.Input(shape=(512, 512, 1))

    unet = Unet()
    unet(input)
    unet.summary()
    
    compile_methods = Compile()

    unet.compile(
        optimizer=compile_methods.optimizer,
        loss=compile_methods.loss,
        metrics=compile_methods.all_metrics
    )

    history = unet.fit(
        x=train_gen,
        steps_per_epoch=100,
        batch_size=8,
        epochs=15,
        verbose=1
    )

if __name__ == "__main__":
    main()