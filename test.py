import os
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision as tv
from torchvision import models
from torch.autograd import Variable
from time import time
from model.madry_model import WideResNet
from model.resnetdsbn import *
from attack import FastGradientSignUntargeted
from utils import makedirs, create_logger, tensor2cuda, numpy2cuda, evaluate, evaluate_, save_model
from argument import parser, print_args
from plot import plot_AUC
import patch_dataset as patd

class Trainer():
    
    def __init__(self, attack, log_folder):
        # self.args = args
        # self.logger = logger
        self.attack = attack
        self.log_folder = log_folder

    def test(self, model, loader, epsilon, adv_test=False, 
                use_pseudo_label=False, if_AUC=False):
        # adv_test is False, return adv_acc as -1 
        total_acc = 0.0
        num = 0
        total_adv_acc = 0.0
        total_stdloss = 0.0
        total_advloss = 0.0
        t = Variable(torch.Tensor([0.5]).cuda()) # threshold to compute accuracy
        label_list = []
        pred_list = []
        predadv_list = []

        with torch.no_grad():
            for data, label in loader:
                data, label = tensor2cuda(data), tensor2cuda(label)
                model.eval()
                output = model(data)
                std_loss = F.binary_cross_entropy(torch.sigmoid(output), label)
                pred = torch.sigmoid(output)
                out = (pred > t).float()
                te_acc = np.mean(evaluate_(out.cpu().numpy(), label.cpu().numpy()))
                total_acc += te_acc
                total_stdloss += std_loss
                if if_AUC:
                    label_list.append(label.cpu().numpy())
                    pred_list.append(pred.cpu().numpy())
                # num += output.shape[0]
                num += 1

                if adv_test:
                    # use predicted label as target label
                    with torch.enable_grad():
                        adv_data = self.attack.perturb(data, 
                                                       pred if use_pseudo_label else label, 
                                                       'mean', False)
                    model.eval()
                    adv_output = model(adv_data)
                    adv_loss = F.binary_cross_entropy(torch.sigmoid(adv_output), label)
                    adv_pred = torch.sigmoid(adv_output)
                    if if_AUC:
                        predadv_list.append(adv_pred.cpu().numpy())
                    adv_out = (adv_pred > t).float()
                    adv_acc = np.mean(evaluate_(adv_out.cpu().numpy(), label.cpu().numpy()))
                    total_adv_acc += adv_acc
                    total_advloss += adv_loss
                else:
                    total_adv_acc = -num
        if if_AUC:
            pred = np.squeeze(np.array(pred_list))
            predadv = np.squeeze(np.array(predadv_list))
            label = np.squeeze(np.array(label_list))
            np.save(os.path.join(self.log_folder, 'y_predadv_'+str(epsilon)+'.npy'), predadv)
        else:
            return total_acc / num, total_adv_acc / num, total_stdloss / num, total_advloss / num

def main():
    log_folder = './results/plots/robustness/nih_std_eps/'
    makedirs(log_folder)
    model = models.resnet50(pretrained=False)
    # model = resnet50dsbn(pretrained=False)
    num_classes=8
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    if torch.cuda.is_available():
        model = model.cuda()
    
    todo = 'test'
    if todo == 'test': # set 'valid' fold for knee and luna dataset and set 'test' fold for CXR dataset
        eps = np.linspace(0.001, 0.1, num=11)
        for i in range(len(eps)):
            epsilon = eps[i]
            alpha = epsilon / 2
            attack = FastGradientSignUntargeted(model, 
                                        epsilon, 
                                        alpha, 
                                        min_val=0, 
                                        max_val=1, 
                                        max_iters=10, 
                                        _type='linf')
            trainer = Trainer(attack, log_folder)
            te_dataset = patd.PatchDataset(path_to_images='../NIH_Chest/images/',
                                            fold='test',
                                            transform=tv.transforms.Compose([
                                                tv.transforms.Resize(256),
                                                tv.transforms.ToTensor()
                                                ]))
            te_loader = DataLoader(te_dataset, batch_size=1, shuffle=False, num_workers=1)
            checkpoint = torch.load('./checkpoint/cxr/chexpert_std_/checkpoint_best.pth')
            model.load_state_dict(checkpoint)
            trainer.test(model, te_loader, i, adv_test=True, 
                                use_pseudo_label=False, if_AUC=True)
            # print("std acc: %.4f, adv_acc: %.4f" % (std_acc * 100, adv_acc * 100))
    else:
        raise NotImplementedError
    

if __name__ == '__main__':
    main()