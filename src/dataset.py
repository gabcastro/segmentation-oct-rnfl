import os
import cv2
import numpy as np
import tensorflow as tf
from glob import glob

class Dataset():
    """Expand the data to create a dataset"""

    def __init__(self, 
                data_dir):
        self.images, self.masks = self.load_data(data_dir)
        print(f'Loaded {len(self.images)} images, and {len(self.masks)} masks')

    def load_data(self, data_dir):
        images = sorted(glob(os.path.join(data_dir, "images/*.png")))
        masks = sorted(glob(os.path.join(data_dir, "masks/*.png")))

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
        image.set_shape([512, 512, 1])
        mask.set_shape([512, 512, 1])

        return image, mask

    def read_mask(self, path):
        x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        x = cv2.resize(x, (512, 512))
        # print(f"shapeee maskk: {x.shape}")
        x = x / 255.0
        x = np.expand_dims(x, axis=-1)
        x = x.astype(np.float32)

        return x

    def read_image(self, path):
        x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        x = cv2.resize(x, (512, 512))
        # print(f"shapeee: {x.shape}")
        x = x / 255.0
        x = np.expand_dims(x, axis=-1)
        x = x.astype(np.float32)
        
        return x

    