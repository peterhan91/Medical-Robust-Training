import os
import torch
import torchvision as tv
import numpy as np
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn
from utils import makedirs, tensor2cuda, load_model, LabelDict
from argument import parser
from visualization import VanillaBackprop
from model.madry_model import WideResNet
import patch_dataset as patd
import matplotlib.pyplot as plt 
from skimage.color import rgb2gray
from model.resnetdsbn import *

args = parser()
img_folder = 'grad_img'
img_folder = os.path.join(img_folder, args.dataset, args.affix)
makedirs(img_folder)
out_num = 1

transform_test = tv.transforms.Compose([
        tv.transforms.Resize(256),
        # tv.transforms.CenterCrop(224),
        tv.transforms.ToTensor(),
        # tv.transforms.Normalize(mean, std)
        ])
te_dataset = patd.PatchDataset(path_to_images=args.data_root, fold='test',
                               transform=tv.transforms.ToTensor())
te_loader = DataLoader(te_dataset, batch_size=1, shuffle=False, num_workers=1)


counter = 0
input_list = []
grad_list = []
label_list = []
for data, label in te_loader:
    # print(np.nonzero(label.squeeze().numpy())[0])
    # disease_index = np.nonzero(label.squeeze().numpy())[0][0]
    # print('label ', label)
    # print('disease index ', disease_index)
    # disease = te_dataset.PRED_LABEL[disease_index]
    if int(np.sum(label.squeeze().numpy())) > 0:
        disease = ''
        for i in range(int(np.sum(label.squeeze().numpy()))):
            disease_index = np.nonzero(label.squeeze().numpy())[0][i]
            dis_temp = te_dataset.PRED_LABEL[disease_index]
            disease = disease + ' ' + dis_temp

        data, label = tensor2cuda(data), tensor2cuda(label)
        model_bns = resnet50dsbn(pretrained=args.pretrain, widefactor=args.widefactor)
        model_std = models.resnet50(pretrained=args.pretrain)
        num_classes=3
        model_bns.fc = nn.Linear(model_bns.fc.in_features, num_classes)
        model_std.fc = nn.Linear(model_std.fc.in_features, num_classes)
        # load_model(model, args.load_checkpoint)
        load_model(model_bns, './checkpoint/rijeka/knee_linf_/checkpoint_best.pth')
        load_model(model_std, './checkpoint/rijeka/knee_std_/checkpoint_best.pth')
        if torch.cuda.is_available():
            model_bns.cuda()
            model_std.cuda()

        VBP = VanillaBackprop(model_bns)
        VBP_std = VanillaBackprop(model_std)
        grad_bn0 = VBP.generate_gradients(data, label, [0]) # data: (1,3,96,96) label: (1,3)
        grad_bn1 = VBP.generate_gradients(data, label, [1])
        grad_std = VBP_std.generate_gradients(data, label)
        grads = []
        # print(grad.shape)
        for grad in [grad_std, grad_bn0, grad_bn1]:
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
        out_list = [data, grad]
        # print('data shape ', data.shape)
        # print('grad shape ', grad.shape)
        input_list.append(data)
        label_list.append(disease)
        grad_list.append(grads)
np.save(os.path.join(img_folder, 'data.npy'), np.array(input_list))
np.save(os.path.join(img_folder, 'label.npy'), np.array(label_list))
np.save(os.path.join(img_folder, 'grad.npy'), np.array(grad_list))
'''
        types = ['Original', 'Your Model']
        fig, _axs = plt.subplots(nrows=len(out_list), ncols=out_num)
        axs = _axs
        for j, _type in enumerate(types):
            axs[j].set_ylabel(_type)
            # if j == 0:
            #     cmap = 'gray'
            # else:
            #     cmap = 'seismic'
            for i in range(out_num):
                # axs[j, i].set_xlabel('%s' % label_dict.label2class(label[i]))
                axs[j].set_xlabel('%s' % disease)
                img = out_list[j]
                # print(img.shape)
                img = np.transpose(img, (1, 2, 0))
                img = img.astype(np.uint8)
                # img_gray = rgb2gray(img)
                axs[j].imshow(img)
                axs[j].get_xaxis().set_ticks([])
                axs[j].get_yaxis().set_ticks([])
                
        plt.tight_layout()
        plt.savefig(os.path.join(img_folder, 'grad_%s_%d.png' % (args.affix, counter)))
        plt.close(fig)
        counter += 1
'''