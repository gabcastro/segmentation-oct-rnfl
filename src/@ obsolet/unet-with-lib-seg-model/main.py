import os
import matplotlib.pyplot as plt
import numpy as np
import cv2

import constants as constants
import dataset as ds
import utility
import augmentation
import dataloader as dl

import segmentation_models as sm

sm.set_framework('tf.keras')

def main():
    x_train_dir = constants.TRAIN_DIR
    y_train_dir = constants.TRAIN_ANNOTATION_DIR

    x_valid_dir = constants.VAL_DIR 
    y_valid_dir = constants.VAL_ANNOTATION_DIR

    x_test_dir = constants.TEST_DIR
    y_test_dir = constants.TEST_ANNOTATION_DIR

    util = utility.Utility

    aug = augmentation.Augmentation()

    BACKBONE = 'resnet34'
    BATCH_SIZE = 8
    CLASSES = ['layer_1']
    EPOCHS = 1
    LR = 0.0001
    preprocess_input = sm.get_preprocessing(BACKBONE)

    test_dataset = ds.Dataset(
        x_test_dir, 
        y_test_dir, 
        classes=CLASSES, 
        augmentation=aug.get_validation_augmentation(),
        preprocessing=aug.get_preprocessing(preprocess_input),
    )

    test_dataloader = dl.Dataloder(test_dataset, batch_size=1, shuffle=False)
    
    test_dataloader[0][0]

    # n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation
    # activation = 'sigmoid' if n_classes == 1 else 'softmax'

    # # create model
    # model = sm.Unet(BACKBONE, classes=n_classes, activation=activation)

    # import tensorflow as tf

    # # define optomizer
    # optim = tf.keras.optimizers.Adam(LR)

    # # Segmentation models losses can be combined together by '+' and scaled by integer or float factor
    # dice_loss = sm.losses.DiceLoss()
    # focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
    # total_loss = dice_loss + (1 * focal_loss)

    # # actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
    # # total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss 

    # metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

    # # compile keras model with defined optimozer, loss and metrics
    # model.compile(optim, total_loss, metrics)

    # train_dataset = ds.Dataset(
    #     x_train_dir, 
    #     y_train_dir, 
    #     classes=CLASSES, 
    #     augmentation=aug.get_training_augmentation(),
    #     preprocessing=aug.get_preprocessing(preprocess_input),
    # )

    # valid_dataset = ds.Dataset(
    #     x_valid_dir, 
    #     y_valid_dir, 
    #     classes=CLASSES, 
    #     augmentation=aug.get_validation_augmentation(),
    #     preprocessing=aug.get_preprocessing(preprocess_input),
    # )

    # train_dataloader = dl.Dataloder(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # valid_dataloader = dl.Dataloder(valid_dataset, batch_size=1, shuffle=False)
    
    # assert train_dataloader[0][0].shape == (BATCH_SIZE, 512, 512, 3)
    # assert train_dataloader[0][1].shape == (BATCH_SIZE, 512, 512, n_classes)

    # callbacks = [
    #     # tf.keras.callbacks.ModelCheckpoint('./best_model.h5', save_weights_only=True, save_best_only=True, mode='min', verbose=1),
    #     tf.keras.callbacks.ReduceLROnPlateau()
    # ]

    # history = model.fit(
    #     train_dataloader, 
    #     steps_per_epoch=len(train_dataloader), 
    #     epochs=EPOCHS,  
    #     callbacks=callbacks, 
    #     validation_data=valid_dataloader, 
    #     validation_steps=len(valid_dataloader)
    # )


if __name__ == "__main__":
    main()