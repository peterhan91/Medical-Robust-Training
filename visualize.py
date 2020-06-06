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

args = parser()
img_folder = 'grad_img'
img_folder = os.path.join(img_folder, args.dataset, args.affix)
makedirs(img_folder)
out_num = 1
# label_dict = LabelDict(args.dataset)
# te_dataset = tv.datasets.CIFAR10(args.data_root, 
#                                train=False, 
#                                transform=tv.transforms.ToTensor(), 
#                                download=True)
# te_loader = DataLoader(te_dataset, batch_size=args.batch_size, shuffle=False, num_workers=48)

# te_dataset = patd.PatchDataset(path_to_images=args.data_root, fold='valid',
#                                transform=tv.transforms.ToTensor())
# te_loader = DataLoader(te_dataset, batch_size=1, shuffle=False, num_workers=1)

te_loader = DataLoader(MRIDataset(directory=args.data_root, 
                            mode='test', rijeka=True, 
                            transform=tv.transforms.Compose([ToTensor()])), 
                            batch_size=args.batch_size, num_workers=1)

counter = 0
# for data, label in te_loader:
for sample in tr_loader:
    data, label = sample['buffers'], sample['labels']
    disease_index = np.nonzero(label.squeeze().numpy())[0][0]
    # print('label ', label)
    # print('disease index ', disease_index)
    disease = te_dataset.PRED_LABEL[disease_index]

    data, label = tensor2cuda(data), tensor2cuda(label)
    num_classes=3
    resnet = models.resnet34(pretrained=False)
    model = I3ResNet(copy.deepcopy(resnet), num_classes)
    # model = models.resnet34(pretrained=False)
    # model.fc = nn.Linear(model.fc.in_features, num_classes)
    load_model(model, args.load_checkpoint)
    if torch.cuda.is_available():
        model.cuda()

    VBP = VanillaBackprop(model)
    grad = VBP.generate_gradients(data, label) # data: (1,3,96,96) label: (1,3)
    # print(grad.shape)
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
    grad *= 255.0
    label = label.cpu().numpy()
    data = data.cpu().numpy().squeeze()
    data *= 255.0
    out_list = [data, grad]
    # print('data shape ', data.shape)
    # print('grad shape ', grad.shape)

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