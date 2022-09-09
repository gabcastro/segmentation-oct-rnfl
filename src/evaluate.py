import os
import numpy as np
import skimage.io as io
import sklearn.metrics as sm

class Evaluate:
    def __init__(self, directory, folders):
        self.dir_grays = os.path.join(directory, folders[0])
        self.dir_masks = os.path.join(directory, folders[1])
        self.dir_pred = os.path.join(directory, folders[2])

        self.img_grays = self.getcontent(self.dir_grays)
        self.img_masks = self.getcontent(self.dir_masks)
        
        self.prec_total=0
        self.rec_total=0
        self.acc_total=0
        self.IoU_total=0
        self.f1_score_total=0

    def getcontent(self, dir):
        content = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
        content = sorted(content)

        return content

    def labelvisualize(self, num_class, color_dict, img):
        img = img[:,:,0] if len(img.shape) == 3 else img
        img_out = np.zeros(img.shape + (3,))
        for i in range(num_class):
            img_out[img == i] = color_dict[i]
      
        return img_out

    def saveimgs(self, 
                 model_predict,
                 flag_multi_class = False, 
                 num_class = 2):
        for i, item in enumerate(model_predict):
            img = self.labelvisualize(num_class, COLOR_DICT, item) if flag_multi_class else item[:,:,0]
            img[img > 0.1]=1
            img[img <= 0.1]=0            
            io.imsave(os.path.join(self.dir_pred, 'pred_' + self.img_grays[i]), img)   

    def metric(self, model_predict):
        for idx, img in enumerate(model_predict):
            img = img[:,:,0]
            print(os.path.join(self.dir_masks, self.img_masks[idx]))
            mask = io.imread(os.path.join(self.dir_masks, self.img_masks[idx]), as_gray=True)

            img1 = np.array(((img - np.min(img)) / np.ptp(img)) > 0.1).astype(float)
            msk1 = np.array(((mask - np.min(mask))/ np.ptp(mask)) > 0.1).astype(float) 

            u, v = np.shape(msk1)
            mask_list = np.reshape(msk1, (u*v,))
            predicted_list = np.reshape(img1, (u*v,))

            tn, fp, fn, tp = sm.confusion_matrix(mask_list, predicted_list, labels=[0,1]).ravel()
            tn, fp, fn, tp = np.float64(tn), np.float64(fp), np.float64(fn), np.float64(tp)

            total = tp + fp + fn + tn
            accuracy = (tp + tn) / total
            prec = tp / (tp + fp)
            rec = tp / (tp + fn)
            IoU = tp / (tp + fp + fn)
            f1_score = (2 * tp) / ((2 * tp) + fp + fn)

            self.acc_total = self.acc_total + accuracy
            self.prec_total = self.prec_total + prec 
            self.rec_total = self.rec_total + rec 
            self.IoU_total = self.IoU_total + IoU
            self.f1_score_total = self.f1_score_total + f1_score

    def summary(self):
        print(
            "\n=> Precision \t : \t", self.prec_total/len(self.img_masks), 
            "\n=> Recall \t : \t", self.rec_total/len(self.img_masks), 
            "\n=> IoU \t\t : \t", self.IoU_total/len(self.img_masks), 
            "\n=> Acc \t\t : \t", self.acc_total/len(self.img_masks), 
            "\n=> F1 \t\t : \t", self.f1_score_total/len(self.img_masks)
        )