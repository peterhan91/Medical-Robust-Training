import os
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, sampler
import torchvision as tv
from torchvision import models
from torch.autograd import Variable

from time import time
from attack import FastGradientSignUntargeted
from utils import makedirs, create_logger, tensor2cuda, numpy2cuda, evaluate, evaluate_, save_model
from argument import parser, print_args
import patch_dataset as patd
# os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [0,2]))

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

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
        best_loss = 999

        for epoch in range(1, args.max_epoch+1):
            for data, label in tr_loader:
            # for sample in tr_loader:
                # data, label = sample['buffers'], sample['labels']
                # print('data shape is ', data.shape)
                # print('label shape is ', label.shape)
                data, label = tensor2cuda(data), tensor2cuda(label)
                if adv_train:
                    # When training, the adversarial example is created from a random 
                    # close point to the original data point. If in evaluation mode, 
                    # just start from the original data point.
                    adv_data = self.attack.perturb(data, label, 'mean', True)
                    model.train()
                    output = model(adv_data)
                else:
                    model.train()
                    output = model(data)
                loss = F.binary_cross_entropy(torch.sigmoid(output), label)
                opt.zero_grad()
                loss.backward()
                opt.step() 
                t = Variable(torch.Tensor([0.5]).cuda()) # threshold to compute accuracy

                if _iter % args.n_eval_step == 0:
                    t1 = time()
                    if adv_train:
                        with torch.no_grad():
                            model.eval()
                            stand_output = model(data)
                        # pred = torch.max(stand_output, dim=1)[1] # this give us the indices tensor
                        pred = torch.sigmoid(stand_output)
                        out = (pred > t).float()
                        # print(pred.shape)
                        # print(out.shape)
                        # std_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy()) * 100
                        stdacc_list = evaluate_(out.cpu().numpy(), label.cpu().numpy())
                        # print('std accuracy list shape: ', np.array(stdacc_list).shape)

                        # pred = torch.max(output, dim=1)[1]
                        pred = torch.sigmoid(output)
                        out = (pred > t).float()
                        # print(pred)
                        # adv_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy()) * 100
                        advacc_list = evaluate_(out.cpu().numpy(), label.cpu().numpy())
                        # print('adv accuracy list shape: ', np.array(stdacc_list).shape)

                    else:
                        
                        adv_data = self.attack.perturb(data, label, 'mean', False)

                        with torch.no_grad():
                            model.eval()
                            adv_output = model(adv_data)
                            # adv_output = model(adv_data, _eval=True)
                        # pred = torch.max(adv_output, dim=1)[1]
                        pred = torch.sigmoid(adv_output)
                        out = (pred > t).float()
                        # print(label)
                        # print(pred)
                        # adv_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy()) * 100
                        advacc_list = evaluate_(out.cpu().numpy(), label.cpu().numpy())

                        # pred = torch.max(output, dim=1)[1]
                        pred = torch.sigmoid(output)
                        out = (pred > t).float()
                        # print(pred)
                        # std_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy()) * 100
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
            scheduler.step()
            if va_loader is not None:
                t1 = time()
                va_acc, va_adv_acc, va_stdloss, va_advloss = self.test(model, va_loader, True, False)
                va_acc, va_adv_acc = va_acc * 100.0, va_adv_acc * 100.0
                if va_stdloss < best_loss:
                    best_loss = va_stdloss
                    file_name = os.path.join(args.model_folder, 'checkpoint_best.pth')
                    save_model(model, file_name)

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
            # for sample in loader:
                # data, label = sample['buffers'], sample['labels']
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
            np.save(os.path.join(self.args.log_folder, 'y_pred.npy'), pred)
            # np.save(os.path.join(self.args.log_folder, 'y_predadv_'+str(args.epsilon)+'.npy'), predadv)
            np.save(os.path.join(self.args.log_folder, 'y_true.npy'), label)
            
            # PRED_label = ['No Finding', 'Cardiomegaly', 'Edema', 
            #                 'Consolidation', 'Pneumonia', 'Atelectasis',
            #                 'Pneumothorax', 'Pleural Effusion']
            # PRED_label = ['healthy', 'partially injured', 'completely ruptured']
            # PRED_label = ['malignancy']
            # plot_AUC(pred, label, self.args.log_folder, 'auc.png', PRED_label)
            # plot_AUC(predadv, label, self.args.log_folder, 'auc_'+str(args.epsilon)+'.png', PRED_label)
            # np.save('predstd_'+str(args.epsilon)+'.npy', pred)
            # np.save('predstdadv_'+str(args.epsilon)+'.npy', predadv)
            # np.save('labelstd_'+str(args.epsilon)+'.npy', label)
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

    # model = WideResNet(depth=34, num_classes=10, widen_factor=10, dropRate=0.0)
    model = models.resnet50(pretrained=args.pretrain)
    num_classes=8
    # model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    attack = FastGradientSignUntargeted(model, 
                                        args.epsilon, 
                                        args.alpha, 
                                        min_val=0, 
                                        max_val=1, 
                                        max_iters=args.k, 
                                        _type=args.perturbation_type)

    if torch.cuda.is_available():
        model.cuda()
        # model = nn.DataParallel(model).cuda()

    trainer = Trainer(args, logger, attack)

    if args.todo == 'train':

        transform_train = tv.transforms.Compose([
                tv.transforms.Resize(256),
                tv.transforms.ToTensor(),
                tv.transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                   (4*6,4*6,4*6,4*6), mode='constant', value=0).squeeze()),
                tv.transforms.ToPILImage(),
                tv.transforms.RandomHorizontalFlip(),
                tv.transforms.ColorJitter(brightness=0.3, contrast=0.3, 
                                            saturation=0.3, hue=0.3),
                # tv.transforms.RandomRotation(25),
                tv.transforms.RandomAffine(25, translate=(0.2, 0.2), 
                                            scale=(0.8,1.2), shear=10),                            
                tv.transforms.RandomCrop(256),
                tv.transforms.ToTensor(),
                AddGaussianNoise(0.5, args.epsilon)
            ])
        tr_dataset = patd.PatchDataset(path_to_images=args.data_root,
                                        fold='train', 
                                        sample=args.subsample,
                                        transform=transform_train)
        tr_loader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True, num_workers=24)
        # evaluation during training
        transform_test = tv.transforms.Compose([
                tv.transforms.Resize(256),
                # tv.transforms.CenterCrop(224),
                tv.transforms.ToTensor(),
                # tv.transforms.Normalize(mean, std)
                ])
        te_dataset = patd.PatchDataset(path_to_images=args.data_root,
                                        fold='valid',
                                        transform=transform_test)
        te_loader = DataLoader(te_dataset, batch_size=args.batch_size, shuffle=False, num_workers=24)
             
        trainer.train(model, tr_loader, te_loader, args.adv_train)
    
    elif args.todo == 'test':
        te_dataset = patd.PatchDataset(path_to_images=args.data_root,
                                        fold='test',
                                        transform=tv.transforms.Compose([
                                            tv.transforms.Resize(256),
                                            tv.transforms.ToTensor(),
                                            ]))
        te_loader = DataLoader(te_dataset, batch_size=1, shuffle=False, num_workers=1)
        checkpoint = torch.load(args.load_checkpoint)
        model.load_state_dict(checkpoint)
        std_acc, adv_acc = trainer.test(model, te_loader, adv_test=True, use_pseudo_label=False, if_AUC=True)
        print("std acc: %.4f, adv_acc: %.4f" % (std_acc * 100, adv_acc * 100))

    else:
        raise NotImplementedError
    

if __name__ == '__main__':
    args = parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    main(args)