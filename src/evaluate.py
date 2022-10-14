import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score

class Evaluate:
    def __init__(self, 
                data_dir,
                save_pred_dir):
        self.images, self.masks = self.load_data(data_dir)
        self.save_pred_dir = save_pred_dir

    def load_data(self, data_dir):
        images = sorted(glob(os.path.join(data_dir, "images/*.png")))
        masks = sorted(glob(os.path.join(data_dir, "masks/*.png")))

        return images, masks

    def save_results(self, image, mask, y_pred, save_dir):
        mask = np.expand_dims(mask, axis=-1)
        mask = np.concatenate([mask, mask, mask], axis=-1)

        y_pred = np.expand_dims(y_pred, axis=-1)
        y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1)
        y_pred = y_pred * 255

        line = np.ones((512, 10, 3)) * 255

        cat_images = np.concatenate([image, line, mask, line, y_pred], axis=1)
        cv2.imwrite(save_dir, cat_images)

    def eval(self, model):
        """Evaluate and save predictions from model trained"""

        SCORE = []
        for img_name, mask_name in tqdm(zip(self.images, self.masks), total=len(self.masks)):
            name = img_name.split("/")[-1]

            image = cv2.imread(img_name, cv2.IMREAD_COLOR)
            img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
            img = img / 255.0
            img = np.expand_dims(img, axis=0)
            img = np.expand_dims(img, axis=-1)
            
            mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)
            
            y_pred = model.predict(img, verbose=2)[0]
            y_pred = np.squeeze(y_pred, axis=-1)
            y_pred = y_pred >= 0.1
            y_pred = y_pred.astype(np.int32)
            
            save_img_dir = os.path.join(self.save_pred_dir, name)
            self.save_results(image, mask, y_pred, save_img_dir)

            mask = mask / 255.0
            mask = (mask > 0.5).astype(np.int32).flatten()
            y_pred = y_pred.flatten()

            f1_value = f1_score(mask, y_pred, labels=[0, 1], average="binary")
            jac_value = jaccard_score(mask, y_pred, labels=[0, 1], average="binary")
            recall_value = recall_score(mask, y_pred, labels=[0, 1], average="binary", zero_division=0)
            precision_value = precision_score(mask, y_pred, labels=[0, 1], average="binary", zero_division=0)
            SCORE.append([name, f1_value, jac_value, recall_value, precision_value])

        score = [s[1:]for s in SCORE]
        score = np.mean(score, axis=0)
        print(f"F1: {score[0]:0.5f}")
        print(f"Jaccard: {score[1]:0.5f}")
        print(f"Recall: {score[2]:0.5f}")
        print(f"Precision: {score[3]:0.5f}")