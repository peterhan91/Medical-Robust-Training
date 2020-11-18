
import os
import torch
import torchvision as tv
from torchvision import models
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from utils import makedirs, tensor2cuda, load_model, LabelDict
from argument import parser
from visualization import VanillaBackprop
from attack import FastGradientSignUntargeted
from model.madry_model import WideResNet
from model.resnetdsbn import *
from model import *
import patch_dataset as patd
import matplotlib.pyplot as plt 

perturbation_type = 'linf'
# out_num = 100
args = parser()
max_epsilon = 0.002
alpha = max_epsilon / 2
save_folder = '%s_%s' % (args.dataset, args.affix)
img_folder = os.path.join(args.log_root, save_folder)
makedirs(img_folder)
args = parser()
# label_dict = LabelDict(args.dataset)

te_dataset = patd.PatchDataset(path_to_images=args.data_root,
                                fold='test',
                                transform=tv.transforms.Compose([
                                            tv.transforms.Resize(256),
                                            tv.transforms.ToTensor(),
                                            ]))
te_loader = DataLoader(te_dataset, batch_size=1, shuffle=True, num_workers=1)

adv_list = []
in_list = []
# model = MLP_bns(input_dim=32*32, output_dim=1)
model = models.resnet50(pretrained=False)
num_classes=8
model.fc = nn.Linear(model.fc.in_features, num_classes)
load_model(model, args.load_checkpoint)
if torch.cuda.is_available():
    model.cuda()
attack = FastGradientSignUntargeted(model, 
                                    max_epsilon, 
                                    alpha, 
                                    min_val=0, 
                                    max_val=1, 
                                    max_iters=args.k, 
                                    _type=perturbation_type)

for data, label in te_loader:
    data, label = tensor2cuda(data), tensor2cuda(label)
    # data = data.view(-1, 32*32)
    # break
    with torch.no_grad():
        adv_data = attack.perturb(data, label, 'mean', False)
        model.eval()
        output = model(adv_data)
        pred = torch.max(output, dim=1)[1]
        adv_list.append(adv_data.cpu().numpy().squeeze())  # (N, 28, 28)
        in_list.append(data.cpu().numpy().squeeze())

# data = data.cpu().numpy().squeeze()  # (N, 28, 28)
# data *= 255.0
# label = label.cpu().numpy()
# adv_list.insert(0, data)
# pred_list.insert(0, label)
print(np.array(adv_list).shape)
print(np.array(in_list).shape)
np.save(os.path.join(img_folder, 'sample_advx.npy'), np.array(adv_list))
np.save(os.path.join(img_folder, 'sample_x.npy'), np.array(in_list))
# print(np.array(pred_list).shape)

'''
types = ['Original', 'Your Model']
fig, _axs = plt.subplots(nrows=len(adv_list), ncols=out_num)
axs = _axs
for j, _type in enumerate(types):
    axs[j, 0].set_ylabel(_type)
    for i in range(out_num):
        # print(pred_list[j][i])
        axs[j, i].set_xlabel('%s' % label_dict.label2class(int(pred_list[j][i])))
        img = adv_list[j][i]
        # print(img.shape)
        img = np.transpose(img, (1, 2, 0))
        img = img.astype(np.uint8)
        axs[j, i].imshow(img)
        axs[j, i].get_xaxis().set_ticks([])
        axs[j, i].get_yaxis().set_ticks([])
plt.tight_layout()
plt.savefig(os.path.join(img_folder, 'Image_large_%s_%s.jpg' % (perturbation_type, args.affix)))
'''