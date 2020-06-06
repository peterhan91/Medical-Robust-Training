import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import operator
import os
import warnings
import random
import imageio
import seaborn as sns
from sklearn import metrics
from skimage.transform import resize
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix
random.seed(20)
sns.set()

def plot_AUC(pred, label, saveDir, PRED_label):
    
    fig, axs = plt.subplots(len(PRED_label), 
                            figsize = (5, 5*len(PRED_label)))
    print('pred list shape ', pred.shape)
    print('label list shape ', label.shape)

    for n, pred_label in enumerate(PRED_label):

        name = PRED_label[n]
        y_true = label[:,n]
        y_pred = pred[:,n]
        print(y_true.shape)
        print(y_pred.shape)
        fpr, tpr, _ = roc_curve(y_true, y_pred)

        axs[n].plot(fpr, tpr, 'b-', alpha = 1, 
                        label = name+'(AUC:%2.2f)' % roc_auc_score(y_true, y_pred))
        axs[n].legend(loc = 4, prop={'size': 8})
        axs[n].set_xlabel('False Positive Rate')
        axs[n].set_ylabel('True Positive Rate')

    fig.savefig(os.path.join(saveDir, 'AUC.png'), dpi=300, bbox_inches = 'tight')