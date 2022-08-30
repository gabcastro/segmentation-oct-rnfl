from keras.preprocessing.image import ImageDataGenerator

class DataGenerator:
    """DataGenerator to expand the image training data, using transformations
    """

    def generetor(self, directory, dirToSave):
        """Generate images with some transformations, like:
            - shift (width and height)
            - zoom
            - horizontal flip

        Args:
            directory: directory of data train
            dirToSave: directory where the generator will save every transformation
        """
        
        data_gen_args = dict(
            width_shift_range=0.05,
            height_shift_range=0.05,
            shear_range=0.1,
            zoom_range=[0.7,1],
            horizontal_flip=True,
            fill_mode='nearest'
        )

        train_datagen = ImageDataGenerator(**data_gen_args)
        mask_datagen = ImageDataGenerator(**data_gen_args)
 
        image_generator = train_datagen.flow_from_directory(
            directory = self.dirLayer,
            classes = ['original_images'],
            class_mode = None,
            color_mode = 'grayscale',
            target_size = self.targetSize,
            batch_size = 8,
            seed = 1,
            save_to_dir = self.saveToDir,
            save_prefix = 'image'
        )

        mask_generator = mask_datagen.flow_from_directory(
            directory = self.dirLayer,
            classes = ['ground_truth'],
            class_mode = None,
            color_mode = 'grayscale',
            target_size = self.targetSize,
            batch_size = 8,
            seed = 1,
            save_to_dir = self.saveToDir,
            save_prefix = 'mask'
        )

        self.train_generator = zip(image_generator, mask_generator)

    def adjustData(self, img, mask):
        """Adjust the data, where: images will be btw 0..255 and masks to range of 0..1 
        """

        img /= 255.
        mask /= 255.
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        return (img, mask)

    def adjustedDataTrain(self):
        for (img, mask) in self.train_generator:
            img, mask = self.adjustData(img, mask)
            yield (img, mask)