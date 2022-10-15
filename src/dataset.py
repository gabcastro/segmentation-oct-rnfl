import os
import cv2
import numpy as np
import tensorflow as tf
from glob import glob
from augment import flip, rotation_zoom, shift

class Dataset:
    """Expand the data to create a dataset"""

    def __init__(self, 
                data_dir,
                augment=False):
        self.images, self.masks = self.load_data(data_dir, "images/*.png", "masks/*.png")

        if (augment):
            self.augment(data_dir)
        
        print(f'Loaded a total of {len(self.images)} images, and {len(self.masks)} masks')
        

    def augment(self, data_dir):
        content = []
        content.append(self.images)
        content.append(self.masks)

        self.create_augmentation(data_dir, content)

        images_aug, masks_aug = self.load_data(data_dir, "images_aug/*.png", "masks_aug/*.png")

        self.images = sorted(self.images + images_aug)
        self.masks = sorted(self.masks + masks_aug)


    def load_data(self, data_dir, folder_img, folder_mask):
        images = sorted(glob(os.path.join(data_dir, folder_img)))
        masks = sorted(glob(os.path.join(data_dir, folder_mask)))

        return images, masks


    def create_augmentation(self, root_dir, lists):
        folders = ["images_aug/", "masks_aug/"]
        ops = []

        for idx, current_list in enumerate(lists):
            folder_dir = os.path.join(root_dir, folders[idx])
            for i in current_list:
                try:
                    name = i.split("/")[-1]

                    img = cv2.imread(i, cv2.IMREAD_COLOR)

                    # operations using cv2 to augment 

                    i_flip = (self.get_img_dir(name, "flip_", folder_dir), flip(img, 1))
                    
                    ops.append(i_flip)
                    ops.append((self.get_img_dir(name, "rotation_l_", folder_dir), rotation_zoom(img, 1.2, -10)))
                    ops.append((self.get_img_dir(name, "rotation_r_", folder_dir), rotation_zoom(img, 1.2, 10)))
                    ops.append((self.get_img_dir(name, "zoom_in_", folder_dir), rotation_zoom(img, 1.3)))
                    ops.append((self.get_img_dir(name, "zoom_in_flip_", folder_dir), rotation_zoom(i_flip[1], 1.3)))
                    ops.append((self.get_img_dir(name, "shift_l_", folder_dir), shift(img, -35, 0)))
                    ops.append((self.get_img_dir(name, "shift_r_", folder_dir), shift(img, 35, 0)))
                    ops.append((self.get_img_dir(name, "shift_flip_l_", folder_dir), shift(i_flip[1], -35, 0)))
                    ops.append((self.get_img_dir(name, "shift_flip_r_", folder_dir), shift(i_flip[1], 35, 0)))

                    self.save_augmented_img(ops)
                except:
                    print(f"something wrong at {i}")
                

    def get_img_dir(self, name, prefix, folder_dir):
        name = prefix + name
        i_dir = os.path.join(folder_dir, name)

        return i_dir

    def save_augmented_img(self, content):
        """A list of tuple obj that contain image name and image"""
        for c in content:
            cv2.imwrite(c[0], c[1])

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
        x = x / 255.0
        x = np.expand_dims(x, axis=-1)
        x = x.astype(np.float32)

        return x

    def read_image(self, path):
        x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        x = cv2.resize(x, (512, 512))
        x = x / 255.0
        x = np.expand_dims(x, axis=-1)
        x = x.astype(np.float32)
        
        return x