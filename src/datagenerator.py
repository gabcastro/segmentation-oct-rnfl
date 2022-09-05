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
    """

    def __init__(self, data_gen_args: dict):
        self.data_gen_args = data_gen_args

    def generetor(self, 
                  directory, 
                  folders, 
                  dirToSave,
                  batch_size = 8,
                  target_size = (512, 512)):
        """Generate images with some transformations, like:
            - shift (width and height)
            - zoom
            - horizontal flip

        Args:
            directory: a directory that contains the folders with BW and mask images
            folders: used as 'classes' in `train_datagen.flow_from_directory`
                expect an array with two values: first element for image generator; 
                second element for mask generator
            dirToSave: directory where the generator will save every transformation
            batch_size: number of batchs
            target_size: Tuple of integers `(height, width)`
        """

        train_datagen = ImageDataGenerator(**self.data_gen_args)
        mask_datagen = ImageDataGenerator(**self.data_gen_args)
 
        image_generator = train_datagen.flow_from_directory(
            directory = directory,
            classes = [folders[0]],
            class_mode = None,
            color_mode = 'grayscale',
            target_size = target_size,
            batch_size = batch_size,
            seed = 1,
            save_to_dir = dirToSave,
            save_prefix = 'image'
        )

        mask_generator = mask_datagen.flow_from_directory(
            directory = directory,
            classes = [folders[1]],
            class_mode = None,
            color_mode = 'grayscale',
            target_size = target_size,
            batch_size = batch_size,
            seed = 1,
            save_to_dir = dirToSave,
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