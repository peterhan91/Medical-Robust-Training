import os
from skimage.transform import rotate
from skimage import exposure
import cv2
import random
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

digits = re.compile(r'(\d+)')
def tokenize(filename):
    return tuple(int(token) if match else token
                for token, match in
                ((fragment, digits.search(fragment))
                for fragment in digits.split(filename)))

class MRIDataset(Dataset):
    def __init__(self, directory, mode='train', clip_len=16, rijeka=False, transform=None):
        self.mode = mode
        self.transform = transform
        self.folder = os.path.join(directory, self.mode)  # get the directory of the specified split
        self.rijeka = rijeka
        if self.rijeka:
            self.views = ['data', 'data', 'data']
            self.df = pd.read_csv('rijeka_acl.csv')
            self.PRED_LABEL = ['ACL tear'] 
            self.scanlists = os.listdir(self.folder)

        else:
            self.views = ['sagittal', 'coronal', 'axial']
            # self.views = ['sagittal', 'sagittal', 'sagittal']
            self.df = pd.read_csv('../MRKnee/Script/i3d/MRnet_labels.csv')
            self.PRED_LABEL = ['abnormality', 'ACL tear', 'meniscal tear']
            self.scanlists = os.listdir(os.path.join(self.folder, self.views[0])) 

        self.scanlists.sort(key=tokenize)
        self.clip_len = clip_len
        self.clip_test = 16
        self.resize_height = 256
        self.resize_width = 256
        self.crop_size = 224
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.df = self.df[self.df['fold'] == self.mode]
        self.df = self.df.set_index('Scan Index')   
        
    def __len__(self):
        return len(self.scanlists)
    
    def loadscans(self, scanname, view_ID):
        scan = np.load(scanname)
        if scan.ndim != 3:
            print('Not 3D input scan numpy array!!!, it has a shape of', scan.shape)
        slice_count = int(scan.shape[0])
        slice_width = int(scan.shape[1])
        slice_height = int(scan.shape[2])
        
        count = 0
        if self.mode == 'train': 
            buffer = np.empty((self.clip_len, self.resize_height, self.resize_width), np.dtype('float32')) # buffer shape: [32 or 16, 256, 256]
            if self.clip_len < slice_count:
                seq = random.sample(range(slice_count), self.clip_len)
                seq.sort(key=int)
            else:
                seq = list(range(slice_count))
                while len(seq) < self.clip_len:
                    seq.insert(int(slice_count/2), int(slice_count/2))
        else:
            buffer = np.empty((self.clip_test, self.resize_height, self.resize_width), np.dtype('float32')) # buffer shape: [32, 256, 256]
            if self.clip_test < slice_count:
                seq = list(range(slice_count))
                seq.sort(key=int)
            else:
                seq = list(range(slice_count))
                while len(seq) < self.clip_test:
                    seq.insert(int(slice_count/2), int(slice_count/2))
        
        while count < buffer.shape[0]:
            index = seq[count]
            frame = scan[index]
            if (slice_height != self.resize_height) or (slice_width != self.resize_width):
                frame = cv2.resize(frame, (self.resize_width, self.resize_height))
            # frame = self.norm(frame) # normalize all slices to the range of [0, 1]
            # frame -= self.mean[view_ID]
            # frame /= self.std[view_ID]
            buffer[count] = frame 
            count += 1
        return buffer 
    
    def crop(self, buffer, crop_size):
        # buffer shape: [32 or 16, 256, 256]
        if self.mode=='train':
            height_index = np.random.randint(buffer.shape[1] - crop_size)
            width_index = np.random.randint(buffer.shape[2] - crop_size)
        else: # center crop for valid and test
            height_index = (buffer.shape[1] - crop_size) // 2
            width_index = (buffer.shape[2] - crop_size) // 2

        buffer = buffer[:, height_index:height_index+crop_size,
                            width_index:width_index+crop_size]
        return buffer # buffer shape: [32 or 16, 224, 224]
    
    def flip(self, buffers):
        # buffer shape: [16, 224, 224]
        if self.mode=='train' and random.random() < 0.5:
            for n in range(len(buffers)):
                for m in range(buffers.shape[1]):
                    buffers[n,m,:,:] = np.fliplr(buffers[n,m,:,:])
        return buffers
    
    def rotation(self, buffer, degree):
        if self.mode=='train' and random.random() < 0.5:
            for n in range(len(buffer)):
                buffer[n] = rotate(buffer[n], degree)
        return buffer
    
    def adj_gamma(self, buffers):
        if self.mode=='train' and random.random() < 0.5:
            for n in range(buffers.shape[0]):
                for m in range(buffers.shape[1]):
                    buffers[n,m,:,:] = exposure.adjust_gamma(buffers[n,m,:,:], gamma=random.uniform(0.5, 1.5))
        return buffers

    def adj_contrast(self, buffers):
        if self.mode=='train' and random.random() < 0.5:
            for n in range(buffers.shape[0]):
                for m in range(buffers.shape[1]):
                    p2, p98 = np.percentile(buffers[n,m,:,:], (2, 98))
                    buffers[n,m,:,:] = exposure.rescale_intensity(buffers[n,m,:,:], in_range=(p2, p98))
        return buffers
    
    def log_correct(self, buffers):
        if self.mode=='train' and random.random() < 0.5:
            for n in range(buffers.shape[0]):
                for m in range(buffers.shape[1]):
                    buffers[n,m,:,:] = exposure.adjust_log(buffers[n,m,:,:])
        return buffers

    def hist_equ(self, buffers):
        if self.mode=='train' and random.random() < 0.5:
            for n in range(buffers.shape[0]):
                for m in range(buffers.shape[1]):
                    buffers[n,m,:,:] = exposure.equalize_hist(buffers[n,m,:,:])
        return buffers

    def norm(self, frame):
        frame_zero = frame - np.amin(frame)
        return frame_zero / (np.amax(frame_zero)+1e-12)

    def norm_vol(self, buffers):
        for n in range(buffers.shape[0]):
            for m in range(buffers.shape[1]):
                # buffers[n,m,:,:] = (self.norm(buffers[n,m,:,:]) - self.mean[n]) / self.std[n]
                buffers[n,m,:,:] = self.norm(buffers[n,m,:,:])
        return buffers

    def __getitem__(self, index):
        buffers = []
        for view_ID, view in enumerate(self.views):
            if self.rijeka:
                name = os.path.join(self.folder, self.scanlists[index])
            else:
                name = os.path.join(self.folder, view, self.scanlists[index])
            buffer = self.loadscans(name, view_ID)
            # print(buffer.shape) # debug
            buffer = self.crop(buffer, self.crop_size) # shape [16, 224, 224]
            # buffer = self.rotation(buffer, random.uniform(-15, 15))
            buffers.append(buffer)
        buffers = np.array(buffers)
        # buffers = self.flip(buffers)
        # buffers = self.log_correct(buffers)
        # buffers = self.adj_gamma(buffers)
        # buffers = self.hist_equ(buffers)
        # buffers = self.adj_contrast(buffers)
        buffers = self.norm_vol(buffers)

        labels = np.zeros(len(self.PRED_LABEL), dtype=int) # one-hot like vector
        for i in range(0, len(self.PRED_LABEL)):
            if(self.df[self.PRED_LABEL[i].strip()].loc[self.scanlists[index]].astype('int') > 0):
            # df.series.str.strip: remove leading and traling characters
                labels[i] = self.df[self.PRED_LABEL[i].strip()].loc[self.scanlists[index]].astype('int')
        sample = {'buffers': buffers, 'labels': labels}
        # print('debug scan name:', self.scanlists[index], 'scan label:', labels)

        if self.transform:
            sample = self.transform(sample)

        return sample 

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        buffers, labels = sample['buffers'], sample['labels']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        return {'buffers': torch.from_numpy(buffers),
                'labels': torch.from_numpy(labels)}


