import os
import cv2
from glob import glob
from operations import *

class Augment:
    """Expand the data to create a dataset"""

    def __init__(self,
                data_dir):
        self.images, self.masks = self.load_data(data_dir, "images/*.png", "masks/*.png")

        self.augment(data_dir)

    def load_data(self, data_dir, folder_img, folder_mask):
        images = sorted(glob(os.path.join(data_dir, folder_img)))
        masks = sorted(glob(os.path.join(data_dir, folder_mask)))

        return images, masks

    def augment(self, data_dir):
        content = []
        content.append(self.images)
        content.append(self.masks)

        self.create_augmentation(data_dir, content)

    def create_augmentation(self, root_dir, lists):
        """Create augmentation to image and mask from train directory
        
        Args:
            - root_dir: The root dir where contains image and mask folder
            - lists: A list with two list inside: images and masks
        """
        folders = ["images_aug/", "masks_aug/"]
        ops = []

        for idx, current_list in enumerate(lists):
            folder_dir = os.path.join(root_dir, folders[idx])
            for i in current_list:
                try:
                    name = i.split("/")[-1]

                    print(f'===> creating operations to image {name}')

                    img = cv2.imread(i, cv2.IMREAD_COLOR)

                    # operations using cv2 to augment 
                    # tuple with image name and image

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

                    # pre-processing is about change color/iluminance, masks won't change
                    if idx == 0:
                        ops.append((self.get_img_dir(name, "ilumi_normalize_", folder_dir), ilumi_normalization(img)))
                    else:
                        ops.append((self.get_img_dir(name, "ilumi_normalize_", folder_dir), img))

                    self.save_augmented_img(ops)
                except:
                    print(f"something get wrong at {i}")

                ops.clear()

    def get_img_dir(self, name, prefix, folder_dir):
        name = prefix + name
        i_dir = os.path.join(folder_dir, name)

        return i_dir

    def save_augmented_img(self, content):
        """A list of tuple obj that contain image name and image"""
        for c in content:
            cv2.imwrite(c[0], c[1])
            print(f'image: {c[0]} saved...')