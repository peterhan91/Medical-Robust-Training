import os
import torch
# import torchvision as tv
import numpy as np
import pickle
# from torch.utils.data import DataLoader
# from torchvision import models
# import torch.nn as nn
from utils import *
# from argument import parser
# from visualization import VanillaBackprop
# from model.dsbn import DomainSpecificBatchNorm2d
# import patch_dataset as patd
# import matplotlib.pyplot as plt 
# from skimage.color import rgb2gray
# from model.resnetdsbn import TwoInputSequential, resnet50dsbn

def hook_fn(m, i, o):
    try: 
        visualisation[m] = o.cpu().numpy()
    except AttributeError:
        visualisation[m] = o[0].cpu().numpy()
        
if __name__=='__main__':
    with open('./results/cka/act_std.pkl', 'rb') as file:
        act_std = pickle.load(file)
        file.close()
    file = open('./results/cka/act_adv.pkl', 'rb')
    act_adv = pickle.load(file)
    file.close()
    file = open('./results/cka/act_bn0.pkl', 'rb')
    act_bn0 = pickle.load(file)
    file.close()
    file = open('./results/cka/act_bn1.pkl', 'rb')
    act_bn1 = pickle.load(file)
    file.close()

    ckas_sa = np.zeros((len(list(act_std.values())), len(list(act_std.values()))))
    ckas_self = np.zeros((len(list(act_std.values())), len(list(act_std.values()))))
    ckas_aself = np.zeros((len(list(act_std.values())), len(list(act_std.values()))))
    ckas_bns = np.zeros((len(list(act_std.values())), len(list(act_std.values()))))
    ckas_bns_ = np.zeros((len(list(act_std.values())), len(list(act_std.values()))))
    ckas_bna = np.zeros((len(list(act_std.values())), len(list(act_std.values()))))
    ckas_bna_ = np.zeros((len(list(act_std.values())), len(list(act_std.values()))))
    ckas_bnsa = np.zeros((len(list(act_std.values())), len(list(act_std.values()))))
    
    assert len(list(act_std.values())) == len(list(act_bn0.values()))
    ckas = []
    for i in range(len(list(act_std.values()))):
        for j in range(len(list(act_std.values()))):
            X_s = list(act_std.values())[i].reshape(196, -1)
            X_a_ = list(act_adv.values())[i].reshape(196, -1)
            X_s_ = list(act_std.values())[j].reshape(196, -1)
            X_a = list(act_adv.values())[j].reshape(196, -1)

            try:
                X_bn0_ = list(act_bn0.values())[i].reshape(196, -1)
                X_bn0 = list(act_bn0.values())[j].reshape(196, -1)
                X_bn1 = list(act_bn1.values())[j].reshape(196, -1)
            except AttributeError:
                X_bn0 = list(act_bn0.values())[i][0].reshape(196, -1)
                X_bn0 = list(act_bn0.values())[j][0].reshape(196, -1)
                X_bn1 = list(act_bn1.values())[j][0].reshape(196, -1)

            ckas_sa[i][j] = cka(gram_linear(X_s), gram_linear(X_a), debiased=True)
            ckas_self[i][j] = cka(gram_linear(X_s), gram_linear(X_s_), debiased=True)
            ckas_aself[i][j] = cka(gram_linear(X_a_), gram_linear(X_a), debiased=True)
            ckas_bns[i][j] = cka(gram_linear(X_s), gram_linear(X_bn0), debiased=True)
            ckas_bns_[i][j] = cka(gram_linear(X_s), gram_linear(X_bn1), debiased=True)
            ckas_bnsa[i][j] = cka(gram_linear(X_bn0_), gram_linear(X_bn1), debiased=True)
            ckas_bna[i][j] = cka(gram_linear(X_a_), gram_linear(X_bn0), debiased=True)
            ckas_bna_[i][j] = cka(gram_linear(X_a_), gram_linear(X_bn1), debiased=True)
    
    ckas.append(ckas_sa)
    ckas.append(ckas_self)
    ckas.append(ckas_aself)
    ckas.append(ckas_bns)
    ckas.append(ckas_bns_)
    ckas.append(ckas_bna)
    ckas.append(ckas_bna_)
    ckas.append(ckas_bnsa)
    
    np.save('./results/ckas.npy', np.array(ckas))
