# segmentation-oct-rnfl

this repo was created to structure some scripts, where is separate by different responsibilities

`modules`
- augmentation.py: contain a class to use albumentations lib
- dataloader.py: load data from dataset
- dataset.py: read images and apply some methods to creat the dataset

`others`: contains some files to execute some operations, like rename files

## to-do:

    create a script to create a folder "data" with all subfolders used during the process
        - data
            - test
            - test_annotation
            - train
            - train_annotation
            - validation
            - validation_annotation

some zips contais the images used in some trains (folder zips)

## dependencies

- TensorFlow
- Keras
- PIL
- OpenCv
- Numpy
- Matplotlib
- segmentation models (https://github.com/qubvel/segmentation_models)
- albumentations