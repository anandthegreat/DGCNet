import numpy as np
from sklearn.metrics import confusion_matrix

class custom_conf_matrix():
    def __init__(self, lbl,n_class):
        self.lbl = lbl
        self.n_class = n_class
        self.conf_mat = np.zeros((self.n_class, self.n_class))
        
    def update_step(self,truth_lbl,pred_lbl):
        if (truth_lbl == 255).all():
            return
        curr_conf_mat = confusion_matrix(y_true=truth_lbl, y_pred=pred_lbl, labels=self.lbl)
        self.conf_mat += curr_conf_mat

    def compute_mean_iou(self):
        
        intersection = np.diag(self.conf_mat)
        ground_truth_set = self.conf_mat.sum(axis=1)
        predicted_set = self.conf_mat.sum(axis=0)
        union =  ground_truth_set + predicted_set - intersection
        
        print(intersection)
        print(union)

        indices = [8,9,10,11,12,13,14,15,19]
        intersection = np.take(intersection, indices)
        union = np.take(union, indices)

        print(intersection)
        print(union)

        return np.mean(intersection / np.maximum(1.0, union.astype(np.float32)))

    def reset(self):
        self.conf_mat = np.zeros((self.n_class, self.n_class))