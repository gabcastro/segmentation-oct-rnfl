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
    preprocess_input = sm.get_preprocessing(BACKBONE)

    train_dataset = ds.Dataset(
        x_train_dir, 
        y_train_dir, 
        classes=CLASSES, 
        augmentation=aug.get_training_augmentation(),
        preprocessing=aug.get_preprocessing(preprocess_input),
    )

    valid_dataset = ds.Dataset(
        x_valid_dir, 
        y_valid_dir, 
        classes=CLASSES, 
        augmentation=aug.get_validation_augmentation(),
        preprocessing=aug.get_preprocessing(preprocess_input),
    )

    train_dataloader = dl.Dataloder(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation
    
    assert train_dataloader[0][0].shape == (BATCH_SIZE, 512, 512, 3)
    assert train_dataloader[0][1].shape == (BATCH_SIZE, 512, 512, n_classes)


if __name__ == "__main__":
    main()