import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from model import MLP

############################ generate 2D gaussian datasets ############################

means = [[1.2, 0.1], [-1.2, -0.1]]
cov = [[0.1, -0.01], [-0.01, 0.002]]
size = 3000
np.random.seed(0)
Xs = np.concatenate([np.random.multivariate_normal(mean=means[i], cov=cov, size=size) for i in range(2)], axis=0)
Xs = (Xs - np.min(Xs)) / (np.max(Xs) - np.min(Xs)) * 2 - 1
Ys = np.concatenate([np.ones(shape=size) * i for i in range(2)], axis=0)
data_train = np.c_[Xs, Ys]
np.random.shuffle(data_train)
data_train = (data_train[:, :2], data_train[:, 2])

x_data = Variable(torch.Tensor(data_train[0]))
y_data = Variable(torch.Tensor(data_train[1]))

model = MLP(input_dim=2, output_dim=1, hidden_dim=128)
