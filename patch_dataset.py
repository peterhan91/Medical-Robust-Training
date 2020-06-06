import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import os
from PIL import Image
from skimage import exposure
from skimage import util


class PatchDataset(Dataset):

    def __init__(self, path_to_images, fold, sample=0, transform=None):

        self.transform = transform
        self.path_to_images = path_to_images
        self.df = pd.read_csv("./label/patch_label.csv")
        self.fold = fold
        # the 'fold' column says something regarding the train/valid/test seperation
        self.df = self.df[self.df['fold'] == fold]
        if(sample > 0 and sample < len(self.df)):
            self.df = self.df.sample(sample, random_state=6)
        # self.df = self.df.set_index("Path")
        self.df = self.df.set_index("scan index")
        # df.set_index: set the dataframe index using existing columns. 
        self.PRED_LABEL = ['healthy', 'partially injured', 'completely ruptured']
        # self.PRED_LABEL = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
        #                     'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
        #                     'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
        #                     'Support Devices']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = Image.open(
            os.path.join(self.path_to_images, self.fold, self.df.index[idx]))
        image = image.convert('RGB')
        label = np.zeros(len(self.PRED_LABEL), dtype=int)
        for i in range(0, len(self.PRED_LABEL)):
             # can leave zero if zero, else make one
            if(self.df[self.PRED_LABEL[i].strip()].iloc[idx].astype('int') > 0):
                # df.series.str.strip: remove leading and traling characters
                label[i] = self.df[self.PRED_LABEL[i].strip()].iloc[idx].astype('int')
                # Becareful with the 'int' type here !!!
        if self.transform:
            image = self.transform(image)

        return (image, label)