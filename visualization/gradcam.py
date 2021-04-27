import sys
sys.path.append("..")

import argparse
from pathlib import Path
import cv2
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision as tv
from torchvision import models, transforms
from torch.utils.data import DataLoader
from utils import load_model, tensor2cuda
import patch_dataset as patd

class FeatureExtractor():
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x

class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        for name, module in self.model._modules.items():
            if module == self.feature_module:
                target_activations, x = self.feature_extractor(x)
            elif "avgpool" in name.lower():
                x = module(x)
                x = x.view(x.size(0),-1)
            else:
                x = module(x)

        return target_activations, x

def preprocess_image(img):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    preprocessing = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    return preprocessing(img.copy()).unsqueeze(0)

class GradCam:
    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, input_img):
        return self.model(input_img)

    def __call__(self, input_img, target_category=None, categories=None):
        if self.cuda:
            input_img = input_img.cuda()

        features, output = self.extractor(input_img)

        if target_category == None:
            target_category = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][target_category] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = one_hot.cuda()
        
        one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, input_img.shape[2:])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam, categories[target_category]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image_path', type=str, default='./examples/both.png',
                        help='Input image path')
    parser.add_argument('--nclass', type=int, default=8,
                        help='Number of classes')
    parser.add_argument('--resolution', type=int, default=256,
                        help='Resolution of images')
    parser.add_argument('--checkpoint_path', type=str, default='',
                        help='Checkpoint path')
    parser.add_argument('--checkpoint_path_', type=str, default='',
                        help='Checkpoint (adv.) path')
    parser.add_argument('--save_path', type=str, default='',
                        help='Checkpoint path')
    parser.add_argument('--dataset', type=str, default='cxr',
                        help='Dataset to use')                        

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_MAGMA)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.moveaxis(np.float32(img.cpu().squeeze()), 0, -1)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


if __name__ == '__main__':
    """ python grad_cam.py <path_to_image>
    1. Loads an image with opencv.
    2. Preprocesses it for VGG19 and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization. """

    args = get_args()
    Path(args.save_path).mkdir(parents=True, exist_ok=True)

    model = models.resnet50()
    model_ = models.resnet50()
    model.fc = nn.Linear(model.fc.in_features, args.nclass)
    model_.fc = nn.Linear(model_.fc.in_features, args.nclass)
    load_model(model, args.checkpoint_path)
    load_model(model_, args.checkpoint_path_)
    grad_cam = GradCam(model=model, feature_module=model.layer4, \
                       target_layer_names=["2"], use_cuda=args.use_cuda)
    grad_cam_ = GradCam(model=model_, feature_module=model_.layer4, \
                       target_layer_names=["2"], use_cuda=args.use_cuda)

    transform_test = tv.transforms.Compose([
            tv.transforms.Resize(args.resolution),
            tv.transforms.ToTensor()
            ])
    te_dataset = patd.PatchDataset(path_to_images=args.image_path, fold='test',
                                transform=tv.transforms.ToTensor())
    te_loader = DataLoader(te_dataset, batch_size=1, shuffle=False, num_workers=1)

    counter = 0
    for data, label in te_loader:
        if int(np.sum(label.squeeze().numpy())) > 0:
            disease = ''
            for i in range(int(np.sum(label.squeeze().numpy()))):
                disease_index = np.nonzero(label.squeeze().numpy())[0][i]
                dis_temp = te_dataset.PRED_LABEL[disease_index]
                disease = disease + ' ' + dis_temp
        data, label = tensor2cuda(data), tensor2cuda(label)
        target_category = None
        grayscale_cam, lab = grad_cam(data, target_category, te_dataset.PRED_LABEL)
        grayscale_cam_, lab_ = grad_cam_(data, target_category, te_dataset.PRED_LABEL)
        # print(grayscale_cam.shape)
        # grayscale_cam = cv2.resize(grayscale_cam, (data.shape[2], data.shape[3]))
        if lab == lab_ and lab != 'No Finding':
            cam = show_cam_on_image(data, grayscale_cam)
            cam_ = show_cam_on_image(data, grayscale_cam_)
            input = np.moveaxis(np.uint8(255 * data.cpu().squeeze().numpy()), 0, -1)
            combi = np.hstack((input, cam, cam_))
            name = 'cam_'+str(counter)+'_'+disease+'_confi_'+lab+'.png'
            cv2.imwrite(os.path.join(args.save_path, name), combi)
            counter += 1
            
        if counter==14:
            print(label)
            break