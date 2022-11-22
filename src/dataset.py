import os
import cv2
import numpy as np
import tensorflow as tf
from glob import glob

class Dataset:
    """Create a dataset using `from_tensor_slices`"""

    def __init__(self, 
                data_dir,
                aug_read=False):
        self.images, self.masks = self.load_data(data_dir, "images/*.png", "masks/*.png")

        if (aug_read):
            images_aug, masks_aug = self.load_data(data_dir, "images_aug/*.png", "masks_aug/*.png")
            self.images = sorted(self.images + images_aug)
            self.masks = sorted(self.masks + masks_aug)

        print(f'Loaded a total of {len(self.images)} images, and {len(self.masks)} masks')
        

    def load_data(self, data_dir, folder_img, folder_mask):
        images = sorted(glob(os.path.join(data_dir, folder_img)))
        masks = sorted(glob(os.path.join(data_dir, folder_mask)))

        return images, masks

    def create_dataset(self, batch=8):
        ds = tf.data.Dataset.from_tensor_slices((self.images, self.masks))
        ds = ds.shuffle(buffer_size=1000)
        ds = ds.map(self.preprocess)
        ds = ds.batch(batch)
        ds = ds.prefetch(2)

        return ds

    def preprocess(self, x, y):
        def f(x, y):
            x = x.decode()
            y = y.decode()

            x = self.read_image(x)
            y = self.read_mask(y)
            
            return x, y

        image, mask = tf.numpy_function(f, [x, y], [tf.float32, tf.float32])
        image.set_shape([640, 640, 1])
        mask.set_shape([640, 640, 1])

        return image, mask

    def read_mask(self, path):
        x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        x = cv2.resize(x, (640, 640))
        x = x / 255.0
        x = np.expand_dims(x, axis=-1)
        x = x.astype(np.float32)

        return x

    def read_image(self, path):
        x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        x = cv2.resize(x, (640, 640))
        x = x / 255.0
        x = np.expand_dims(x, axis=-1)
        x = x.astype(np.float32)
        
        return x