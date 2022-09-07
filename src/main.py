from unet import Unet
from compile import Compile
from datagenerator import DataGenerator
import tensorflow as tf

def main():
    data_gen_args = dict(
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.1,
        zoom_range=[0.7,1],
        horizontal_flip=True,
        fill_mode='nearest'
    )

    datagen = DataGenerator(data_gen_args=data_gen_args,
                            directory='../data/data-gen-L1/train',
                            folders=['grays', 'masks', 'augmentations'])
    transformations = datagen()
    
    shape=(512, 512, 1)
    input = tf.keras.Input(shape=shape)

    unet = Unet()
    unet(input)
    unet.model(input_shape=shape).summary()

    print('total trainable weights from unet: ', len(unet.trainable_weights))
    
    compile_methods = Compile()

    unet.compile(
        optimizer=compile_methods.optimizer,
        loss=compile_methods.loss,
        metrics=compile_methods.all_metrics
    )

    history = unet.fit(
        x=transformations,
        steps_per_epoch=100,
        batch_size=8,
        epochs=15,
        verbose=1
    )

if __name__ == "__main__":
    main()