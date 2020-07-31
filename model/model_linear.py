import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import count_parameters
from model.dsbn import DomainSpecificBatchNorm2d

class Expression(nn.Module):
    def __init__(self, func):
        super(Expression, self).__init__()
        self.func = func
    
    def forward(self, input):
        return self.func(input)

class SimpleModel(nn.Module):
    def __init__(self, i_c=1, n_c=10, num_domains=2):
        super(SimpleModel, self).__init__()

        self.conv1 = nn.Conv2d(i_c, 32, 5, stride=1, padding=2, bias=True)
        self.bn1 = DomainSpecificBatchNorm2d(32, num_domains)
        self.pool1 = nn.MaxPool2d((2, 2), stride=(2, 2), padding=0)

        self.conv2 = nn.Conv2d(32, 64, 5, stride=1, padding=2, bias=True)
        self.pool2 = nn.MaxPool2d((2, 2), stride=(2, 2), padding=0)


        self.flatten = Expression(lambda tensor: tensor.view(tensor.shape[0], -1))
        self.fc1 = nn.Linear(7 * 7 * 64, 1024, bias=True)
        self.fc2 = nn.Linear(1024, n_c)


    def forward(self, x_i, domain):
            
        x_o = self.conv1(x_i)
        # x_o = torch.relu(x_o)
        x_o, _ = self.bn1(x_o, domain)
        x_o = self.pool1(x_o)

        x_o = self.conv2(x_o)
        # x_o = torch.relu(x_o)
        x_o = self.pool2(x_o)

        x_o = self.flatten(x_o)

        # x_o = torch.relu(self.fc1(x_o))
        x_o = self.fc1(x_o)

        self.train()

        return self.fc2(x_o), domain