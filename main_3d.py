import os
import sys
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from torchvision import models
from torch.autograd import Variable
from time import time
from attack import FastGradientSignUntargeted
from utils import makedirs, create_logger, tensor2cuda, numpy2cuda, evaluate, evaluate_, save_model
from argument import parser, print_args
from plot import plot_AUC
import patch_dataset as patd
from vol_dataset import MRIDataset, ToTensor
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [1,2]))
sys.path.insert(1, '../MRKnee/Script/i3d')
from inflate_src.i3res import I3ResNet

class Trainer():
    
    def __init__(self, args, logger, attack):
        self.args = args
        self.logger = logger
        self.attack = attack

    def train(self, model, tr_loader, va_loader=None, adv_train=False):
        args = self.args
        logger = self.logger
        opt = torch.optim.Adam(model.parameters(), args.learning_rate, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[100, 200], gamma=0.1)
        _iter = 0
        begin_time = time()
        for epoch in range(1, args.max_epoch+1):
            for sample in tr_loader:
                data, label = sample['buffers'], sample['labels']
                # print('data shape: ', data.shape)
                # print('label shape: ', label.shape)
                data, label = tensor2cuda(data), tensor2cuda(label)
                opt.zero_grad()
                if adv_train:
                    adv_data = self.attack.perturb(data, label, 'mean', True)
                    model.train()
                    output = model(adv_data)
                else:
                    model.train()
                    output = model(data)
                loss = F.binary_cross_entropy(torch.sigmoid(output), label)
                loss.backward()
                opt.step()
                scheduler.step()
                
                t = Variable(torch.Tensor([0.5]).cuda()) # threshold to compute accuracy

                if _iter % args.n_eval_step == 0:
                    t1 = time()
                    if adv_train:
                        with torch.no_grad():
                            model.eval()
                            stand_output = model(data)
                        pred = torch.sigmoid(stand_output)
                        out = (pred > t).float()
                        stdacc_list = evaluate_(out.cpu().numpy(), label.cpu().numpy())
                        pred = torch.sigmoid(output)
                        out = (pred > t).float()
                        advacc_list = evaluate_(out.cpu().numpy(), label.cpu().numpy())
                    else:
                        adv_data = self.attack.perturb(data, label, 'mean', False)
                        with torch.no_grad():
                            model.eval()
                            adv_output = model(adv_data)
                        pred = torch.sigmoid(adv_output)
                        out = (pred > t).float()
                        advacc_list = evaluate_(out.cpu().numpy(), label.cpu().numpy())
                        pred = torch.sigmoid(output)
                        out = (pred > t).float()
                        stdacc_list = evaluate_(out.cpu().numpy(), label.cpu().numpy())
                    t2 = time()
                    print('%.3f' % (t2 - t1))
                    logger.info('epoch: %d, iter: %d, spent %.2f s, tr_loss: %.3f' % (
                        epoch, _iter, time()-begin_time, loss.item()))
                    begin_time = time()
                if _iter % args.n_checkpoint_step == 0:
                    file_name = os.path.join(args.model_folder, 'checkpoint_%d.pth' % _iter)
                    save_model(model, file_name)
                _iter += 1

            if va_loader is not None:
                t1 = time()
                va_acc, va_adv_acc, va_stdloss, va_advloss = self.test(model, va_loader, True, False)
                va_acc, va_adv_acc = va_acc * 100.0, va_adv_acc * 100.0
                t2 = time()
                logger.info('\n'+'='*20 +' evaluation at epoch: %d iteration: %d '%(epoch, _iter) \
                    +'='*20)
                logger.info('test acc: %.3f %%, test adv acc: %.3f %%, spent: %.3f' % (
                    va_acc, va_adv_acc, t2-t1))
                logger.info('test loss: %.3f , test adv loss: %.3f , spent: %.3f' % (
                    va_stdloss, va_advloss, t2-t1))
                logger.info('='*28+' end of evaluation '+'='*28+'\n')

    def test(self, model, loader, adv_test=False, 
                use_pseudo_label=False, if_AUC=False):
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
            for sample in loader:
                data, label = sample['buffers'], sample['labels']
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
            PRED_label = ['No Finding', 'Cardiomegaly', 'Edema', 
                            'Consolidation', 'Pneumonia', 'Atelectasis',
                            'Pneumothorax', 'Pleural Effusion']
            plot_AUC(pred, label, self.args.log_folder, 'auc.png', PRED_label)
            plot_AUC(predadv, label, self.args.log_folder, 'auc_'+str(args.epsilon)+'.png', PRED_label)
            np.save('predstd_'+str(args.epsilon)+'.npy', pred)
            np.save('predstdadv_'+str(args.epsilon)+'.npy', predadv)
            np.save('labelstd_'+str(args.epsilon)+'.npy', label)
        else:
            return total_acc / num, total_adv_acc / num, total_stdloss / num, total_advloss / num

def main(args):
    save_folder = '%s_%s' % (args.dataset, args.affix)
    log_folder = os.path.join(args.log_root, save_folder)
    model_folder = os.path.join(args.model_root, save_folder)
    makedirs(log_folder)
    makedirs(model_folder)
    setattr(args, 'log_folder', log_folder)
    setattr(args, 'model_folder', model_folder)
    logger = create_logger(log_folder, args.todo, 'info')
    print_args(args, logger)

    resnet = models.resnet50(pretrained=args.pretrain)
    num_cla=3
    resnet.fc = nn.Linear(resnet.fc.in_features, num_cla)
    model = I3ResNet(copy.deepcopy(resnet), num_cla)
    if torch.cuda.is_available():
        model = nn.DataParallel(model).cuda()

    attack = FastGradientSignUntargeted(model, 
                                        args.epsilon, 
                                        args.alpha, 
                                        min_val=0, 
                                        max_val=1, 
                                        max_iters=args.k, 
                                        _type=args.perturbation_type)

    trainer = Trainer(args, logger, attack)

    if args.todo == 'train':
        MR_dataset = MRIDataset(directory=args.data_root,
                                transform=transforms.Compose([ToTensor()]))
        tr_loader = DataLoader(MR_dataset, batch_size=args.batch_size, 
                                shuffle=True, num_workers=32, pin_memory=True)
        # evaluation during training
        te_loader = DataLoader(MRIDataset(directory=args.data_root, mode='valid',
                                        transform=transforms.Compose([ToTensor()])), 
                                        batch_size=args.batch_size, num_workers=32, 
                                        pin_memory=True)

        trainer.train(model, tr_loader, te_loader, args.adv_train)
    
    elif args.todo == 'test':
        te_dataset = patd.PatchDataset(path_to_images=args.data_root,
                                        fold='test',
                                        transform=transform_test)
        te_loader = DataLoader(te_dataset, batch_size=1, shuffle=False, num_workers=0)
        te_loader = DataLoader(MRIDataset(directory=args.data_root, mode='valid',
                                transform=transforms.Compose([ToTensor()])), 
                                batch_size=1)
        checkpoint = torch.load(args.load_checkpoint)
        model.load_state_dict(checkpoint)
        std_acc, adv_acc = trainer.test(model, te_loader, adv_test=True, use_pseudo_label=False, if_AUC=True)
        print("std acc: %.4f, adv_acc: %.4f" % (std_acc * 100, adv_acc * 100))
    else:
        raise NotImplementedError
    
if __name__ == '__main__':
    args = parser()
    main(args)