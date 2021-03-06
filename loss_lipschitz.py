import os
import torch
import torchvision as tv
import numpy as np
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn
from utils import *
from model.dsbn import *
import patch_dataset as patd
import matplotlib.pyplot as plt 
from model.resnetdsbn import *
from matplotlib.colors import LightSource
from pathlib import Path


def draw_loss(model, X, epsilon, bn, name, dsbn=True, save=False, plot=False):
    Xi, Yi = np.meshgrid(np.linspace(-epsilon, epsilon,14), np.linspace(-epsilon,epsilon,14))
    
    def grad_at_delta(delta):
        delta.requires_grad_(True)
        if dsbn:
            nn.BCELoss()(torch.sigmoid(model(X+delta, bn)), y).backward()
        else:
            nn.BCELoss()(torch.sigmoid(model(X+delta)), y).backward()
        return delta.grad.detach().sign().view(-1).cpu().numpy()

    dir1 = grad_at_delta(torch.zeros_like(X, requires_grad=True))
    delta2 = torch.zeros_like(X, requires_grad=True)
    delta2.data = torch.tensor(dir1).view_as(X).cuda()
    dir2 = grad_at_delta(delta2)
    np.random.seed(0)
    dir2 = np.sign(np.random.randn(dir1.shape[0]))
    
    all_deltas = torch.tensor((np.array([Xi.flatten(), Yi.flatten()]).T @ 
                              np.array([dir2, dir1])).astype(np.float32)).cuda()
    all_deltas = all_deltas.view(-1,3,256,256)
    Zi = torch.zeros(14*14)
    for n, delta in enumerate(all_deltas):
        if dsbn:
            yp = torch.sigmoid(model(delta + X, bn))
        else:
            yp = torch.sigmoid(model(delta + X))
        zi = nn.BCELoss(reduction="mean")(yp, y)
        Zi[n] = zi
    Zi = Zi.reshape(*Xi.shape).detach().cpu().numpy()
    #Zi = (Zi-Zi.min())/(Zi.max() - Zi.min())
    if plot:
        fig = plt.figure(figsize=(10,10))
        ax = fig.gca(projection='3d')
        ls = LightSource(azdeg=0, altdeg=200)
        rgb = ls.shade(Zi, plt.cm.coolwarm)
        ax.set_zlim(0, 1.0)
        surf = ax.plot_surface(Xi, Yi, Zi, rstride=1, cstride=1, lw=0.5,
                        antialiased=True, facecolors=rgb, alpha=0.7)
        surf = ax.contourf(Xi, Yi, Zi, zdir='z', offset=0, cmap=plt.cm.coolwarm)
        ax.set_xlabel(r'$\epsilon_{Rad.}$', fontsize=20)
        ax.set_ylabel(r'$\epsilon_{\nabla_x \mathcal{L}}$', fontsize=20)
        surf.set_clim(0.2, 0.8)
    
    # plt.show()
    if save:
        plt.savefig(name, format='png', dpi=500, bbox_inches = 'tight', pad_inches = 0)
        plt.close()
    
    return Zi


data_root = '../CheXpert_Dataset/images_256/images/'
transform_test = tv.transforms.Compose([
        tv.transforms.Resize(256),
        tv.transforms.ToTensor()
        ])
te_dataset = patd.PatchDataset(path_to_images=data_root, fold='test',
                               transform=transform_test)
test_loader = DataLoader(te_dataset, batch_size=1, shuffle=False, num_workers=1)

# model = models.resnet50(pretrained=False)
model = resnet50dsbn(pretrained=False)
num_classes=8
model.fc = nn.Linear(model.fc.in_features, num_classes)
modelpath = './checkpoint/cxr/chexpert_linf_full_/checkpoint_best.pth'
if os.path.exists(modelpath):
    load_model(model, modelpath)
    print('Model reloaded!!!')
else:
    print('No checkpoint was found!!!')
if torch.cuda.is_available():
            model = model.cuda()

lipschitz = []
for i, (X,y) in enumerate(test_loader):
    X,y = X.cuda().float(), y.cuda().float()
    dir = './loss_surf/dsbn_adv'
    Path(dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(dir, str(i)+'.png') 
    Z = draw_loss(model, X[0:1], 0.5, [1], path, dsbn=True)

    lips = []
    for n in range(len(Z)):
        lst = Z[:, n]
        res = [x-y for x, y in zip(lst, lst[1:])]
        lips.append(res)
    
    lipschitz.append(lips)

np.save('./jupyter/lips_bna.npy', np.array(lipschitz))
