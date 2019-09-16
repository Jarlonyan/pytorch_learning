#coding=utf-8

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision as tv
from torchvision.transforms import ToPILImage
from torch.autograd import Variable
import torchvision.models as models

class Net(nn.Module):
    def __init__(self:
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0,0.1)
        self.out = nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal_(0,0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value

class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0
        self.memor_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES*2+2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=0.001)
        self.loss_func = nn.MSELoss()

