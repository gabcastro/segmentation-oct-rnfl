import os
import tensorflow as tf
from datetime import datetime

from unet import Unet
from compile import Compile
from datagenerator import DataGenerator
from evaluate import Evaluate

import sys
sys.path.insert(1, '/common')

from common.helperviz import Utility

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
    trasnformations = datagen()
    
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

    filepath = f'./bestmodel__{datetime.now().strftime("%Y%m%d-%H%M%S")}.hdf5'
    model_checkpoint = unet.modelcheckpoint(filepath)

    plateau = tf.keras.callbacks.ReduceLROnPlateau()

    history = unet.fit(
        x=datagen.adjustedDataTrain(trasnformations),
        steps_per_epoch=5,
        batch_size=8,
        epochs=1,
        verbose=1,
        callbacks=[model_checkpoint, plateau]
    )

    viz = Utility()

    viz.visualize_metrics(metrics=['dice_coef', 'soft_dice_coef'], 
                          loss=['loss'],
                          model_history=history)


    dir_test_imgs = '../data/data-gen-L1/test/grays'
    test_imgs = [os.path.join(dir_test_imgs, f) for f in os.listdir(dir_test_imgs) if os.path.isfile(os.path.join(dir_test_imgs, f))]

    datagentest = datagen.dataTestGen(test_imgs)

    pred_result = unet.predict(x=datagentest,
                               batch_size=len(test_imgs),
                               verbose=2)

    eval = Evaluate(directory='../data/data-gen-L1/test', 
                    folders=['grays', 'masks', 'predicted'])
    eval.saveimgs(model_predict=pred_result)
    eval.metric(pred_result)
    eval.summary()

if __name__ == "__main__":
    main()