import os
import torch
import torchvision as tv
import numpy as np
import pickle
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn
from utils import *
from argument import parser
from visualization import VanillaBackprop
from model.dsbn import DomainSpecificBatchNorm2d
import patch_dataset as patd
import matplotlib.pyplot as plt 
from skimage.color import rgb2gray
from model.resnetdsbn import TwoInputSequential, resnet50dsbn, Conv2d

args = parser()
transform_test = tv.transforms.Compose([
        tv.transforms.Resize(256),
        tv.transforms.ToTensor()
        ])
te_dataset = patd.PatchDataset(path_to_images=args.data_root, fold='test',
                               transform=tv.transforms.ToTensor())
te_loader = DataLoader(te_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)

num_classes=8
# model = models.resnet50(pretrained=False)
model = resnet50dsbn(pretrained=args.pretrain, widefactor=args.widefactor)
model.fc = nn.Linear(model.fc.in_features, num_classes)
path = './checkpoint/cxr/chexpert_linf_full_/checkpoint_best.pth'

try:
    load_model(model, path)
    print('Loading weights!')
except:
    raise Exception('No checkpoint found! Please recheck!!!')
if torch.cuda.is_available():
    model.cuda()

visualisation = {}
def hook_fn(m, i, o):
    try: 
        visualisation[m] = o.cpu().numpy()
    except AttributeError:
        visualisation[m] = o[0].cpu().numpy()

def get_all_layers(net):
    # for name, layer in net._modules.items():
    #     if isinstance(layer, nn.Sequential):
    #         get_all_layers(layer)
    #     elif isinstance(layer, TwoInputSequential):
    #         get_all_layers(layer)
    #     else:
    #         layer.register_forward_hook(hook_fn)
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, Conv2d):
            layer.register_forward_hook(hook_fn)

with torch.no_grad():
    for data, label in te_loader:
        data, label = tensor2cuda(data), tensor2cuda(label)
        model.eval()
        get_all_layers(model)
        _ = model(data, [1])
        keys = list(visualisation.keys())
        print(len(keys))
        a_file = open('./results/cka/'+args.affix, "wb")
        pickle.dump(visualisation, a_file)
        a_file.close()
        break