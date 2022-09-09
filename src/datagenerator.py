import os
import cv2
import numpy as np
import skimage.io as io
import skimage.transform as trans
from keras.preprocessing.image import ImageDataGenerator

class DataGenerator:
    """Expand the image training data, using transformations
    On contructor must be passed a dictonary with the transformations, e.g.:

    ```python
    data_gen_args = dict(
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.1,
        zoom_range=[0.7,1],
        horizontal_flip=True,
        fill_mode='nearest'
    )
    ```

    Generate images with some transformations, like:
            - shift (width and height)
            - zoom
            - horizontal flip
    """

    def __init__(self, 
                directory,
                folders,
                data_gen_args: dict,
                batch_size = 8,
                target_size = (512, 512)):
        """
        
        Args:
            directory: a directory that contains the folders with BW and mask images
            folders: used as 'classes' in `train_datagen.flow_from_directory`
                expect an array with name of three folders: images in gray scale; 
                masks; and where will be save the transformations 
            batch_size: number of batchs
            target_size: Tuple of integers `(height, width)`
        """
        self.data_gen_args = data_gen_args
        
        self.dirRoot = directory
        self.folderGray = folders[0]
        self.folderMasks = folders[1]
        self.fullDirAugmentation = os.path.join(self.dirRoot, folders[2])

        self.batch_size = batch_size
        self.target_size = target_size

    def __call__(self):

        train_datagen = ImageDataGenerator(**self.data_gen_args)
        mask_datagen = ImageDataGenerator(**self.data_gen_args)
 
        image_generator = train_datagen.flow_from_directory(
            directory = self.dirRoot,
            classes = [self.folderGray],
            class_mode = None,
            color_mode = 'grayscale',
            target_size = self.target_size,
            batch_size = self.batch_size,
            seed = 1,
            save_to_dir = self.fullDirAugmentation,
            save_prefix = 'image'
        )

        mask_generator = mask_datagen.flow_from_directory(
            directory = self.dirRoot,
            classes = [self.folderMasks],
            class_mode = None,
            color_mode = 'grayscale',
            target_size = self.target_size,
            batch_size = self.batch_size,
            seed = 1,
            save_to_dir = self.fullDirAugmentation,
            save_prefix = 'mask'
        )

        return zip(image_generator, mask_generator)

    def adjustData(self, img, mask):
        """Adjust the data, where: images will be btw 0..255 and masks to range of 0..1 
        """
        img /= 255.
        mask /= 255.
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        return (img, mask)

    def adjustedDataTrain(self, train_gen):
        """For each unpack image and mask, adjust the data"""
        for (img, mask) in train_gen:
            img, mask = self.adjustData(img, mask)
            yield (img, mask)

    def dataTestGen(self, 
                    content_imgs,
                    target_size = (512, 512), 
                    flag_multi_class = False):
        """Create a new datagenerator for test"""
        for i in content_imgs:
            img = io.imread(i, as_gray = True)
            img = trans.resize(img, target_size)
            img = np.reshape(img, img.shape + (1,)) if (not flag_multi_class) else img 
            img = np.reshape(img, (1,) + img.shape)
            yield img