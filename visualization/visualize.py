import sys
sys.path.append("..")

import os
import torch
import torchvision as tv
import numpy as np
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn
from utils import makedirs, tensor2cuda, load_model
from argument import parser
from visualization import VanillaBackprop
import patch_dataset as patd
from model.resnetdsbn import *

args = parser()
img_folder = 'grad_img'
img_folder = os.path.join(img_folder, args.dataset, args.affix)
makedirs(img_folder)
out_num = 1

transform_test = tv.transforms.Compose([
        tv.transforms.Resize(256),
        tv.transforms.ToTensor()
        ])
te_dataset = patd.PatchDataset(path_to_images=args.data_root, fold='test',
                               transform=tv.transforms.ToTensor())
te_loader = DataLoader(te_dataset, batch_size=1, shuffle=False, num_workers=1)


counter = 0
input_list = []
grad_list = []
label_list = []
for data, label in te_loader:
    if int(np.sum(label.squeeze().numpy())) > 0:
        disease = ''
        for i in range(int(np.sum(label.squeeze().numpy()))):
            disease_index = np.nonzero(label.squeeze().numpy())[0][i]
            dis_temp = te_dataset.PRED_LABEL[disease_index]
            disease = disease + ' ' + dis_temp

        data, label = tensor2cuda(data), tensor2cuda(label)
        # model_bns = resnet50dsbn(pretrained=args.pretrain, widefactor=args.widefactor)
        model_std = models.resnet50()
        num_classes=8
        # model_bns.fc = nn.Linear(model_bns.fc.in_features, num_classes)
        model_std.fc = nn.Linear(model_std.fc.in_features, num_classes)
        # load_model(model_bns, args.load_checkpoint)
        load_model(model_std, '../checkpoint/chexpert_gaussn_0.1/checkpoint_best.pth')
        if torch.cuda.is_available():
            # model_bns.cuda()
            model_std.cuda()

        # VBP = VanillaBackprop(model_bns)
        VBP_std = VanillaBackprop(model_std)
        # grad_bn0 = VBP.generate_gradients(data, label, [0]) # data: (1,3,96,96) label: (1,3)
        # grad_bn1 = VBP.generate_gradients(data, label, [1])
        grad_std = VBP_std.generate_gradients(data, label)
        grads = []
        # print(grad.shape)
        for grad in [grad_std]:
            grad_flat = grad.view(grad.shape[0], -1) # grad: (1, 3x96x96)
            mean = grad_flat.mean(1, keepdim=True).unsqueeze(2).unsqueeze(3) # (1,1,1,1)
            std = grad_flat.std(1, keepdim=True).unsqueeze(2).unsqueeze(3) # (1,1,1,1)
            mean = mean.repeat(1, 1, data.shape[2], data.shape[3])
            std = std.repeat(1, 1, data.shape[2], data.shape[3])
            grad = torch.max(torch.min(grad, mean+3*std), mean-3*std)
            print(grad.min(), grad.max())
            grad -= grad.min()
            grad /= grad.max()
            grad = grad.cpu().numpy().squeeze()  # (N, 28, 28)
            grads.append(grad)
        # grad *= 255.0
        # label = label.cpu().numpy()
        data = data.cpu().numpy().squeeze()
        # data *= 255.0
        # print('data shape ', data.shape)
        # print('grad shape ', grad.shape)
        input_list.append(data)
        label_list.append(disease)
        grad_list.append(grads)
# np.save(os.path.join(img_folder, 'data.npy'), np.array(input_list))
np.save(os.path.join(img_folder, 'label.npy'), np.array(label_list))
np.save(os.path.join(img_folder, 'grad.npy'), np.array(grad_list))