import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers

AUTOTUNE = tf.data.AUTOTUNE

class DataGenerator:
    """Expand the image training data, using transformations"""

    def __init__(self, 
                data_imgs_dir,
                data_masks_dir,
                batch_size = 32,
                target_size = (512, 512)):
        
        self.data_imgs_dir = data_imgs_dir
        self.data_masks_dir = data_masks_dir

        self.batch_size = batch_size
        self.target_size = target_size

        self.data_augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.2),
            layers.RandomZoom(.2, .2),
        ])

    def __call__(self):
        train_ds_imgs, val_ds_imgs = self.create_dataset_from_directory(self.data_imgs_dir)
        train_ds_masks, val_ds_masks = self.create_dataset_from_directory(self.data_masks_dir)

        train_ds_imgs = self.normalization_ds(train_ds_imgs)
        val_ds_imgs = self.normalization_ds(val_ds_imgs)
        train_ds_masks = self.normalization_ds(train_ds_masks)
        val_ds_masks = self.normalization_ds(val_ds_masks)

        train_ds_imgs = self.prepere(train_ds_imgs, shuffle=False, augment=True)
        val_ds_imgs = self.prepere(val_ds_imgs)
        train_ds_masks = self.prepere(train_ds_masks, shuffle=False, augment=True)
        val_ds_masks = self.prepere(val_ds_masks)

        return train_ds_imgs, val_ds_imgs, train_ds_masks, val_ds_masks

    def prepere(self, ds, shuffle=False, augment=False):
        """Config ds to use data augmentation and performance on I/O ops

        https://www.tensorflow.org/tutorials/images/data_augmentation#apply_the_preprocessing_layers_to_the_datasets
        https://www.tensorflow.org/tutorials/load_data/images#configure_the_dataset_for_performance

        Args:
            ds: tf.data.Dataset
        """
        if shuffle:
            ds = ds.shuffle(1000)

        # Batch all ds
        # ds = ds.batch(self.batch_size)    

        # Use data augmentation only on the training set
        if augment:
            ds = ds.map(lambda x, y: (self.data_augmentation(x, training=True), y),
                        num_parallel_calls=AUTOTUNE)

        # Use buffered prefetching on all dataset
        return ds.cache().prefetch(buffer_size=AUTOTUNE)

    def normalization_ds(self, ds):
        """The RGB channel values are in the [0, 255] range. 
        This is not ideal for a neural network. In general you should seek to make your input values small.

        Args:
            ds: tf.data.Dataset
        """
        normalization_layer = tf.keras.layers.Rescaling(1./255)
        normalized_ds = ds.map(lambda x, y: (normalization_layer(x), y))

        return normalized_ds
        
    def create_dataset_from_directory(self, data_dir):
        """Generates a tf.data.Dataset from image files in a directory.
        https://www.tensorflow.org/api_docs/python/tf/keras/utils/image_dataset_from_directory

        Args:
            data_dir: directory used (dataset to images or masks)
        """
        train_ds = tf.keras.utils.image_dataset_from_directory(
            directory=data_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=self.target_size,
            batch_size=self.batch_size
        )

        val_ds = tf.keras.utils.image_dataset_from_directory(
            directory=data_dir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=self.target_size,
            batch_size=self.batch_size
        )
        
        return train_ds, val_ds