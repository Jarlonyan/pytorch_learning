#coding=utf-8

import numpy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision as tv
from torchvision.transforms import ToPILImage
from torch.autograd import Variable
import torchvision.models as models

class Net(nn.Module):
    def __init__(self, num=10):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = self.flaten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def flaten(self, x):
        size = x.size()[1:]
        dim = 1
        for s in size:
            dim *= s
        x = x.view(-1, dim)
        return x

class FinetuneNet(nn.Module):
    def __init__(self, num=10):
        super(FinetuneNet, self).__init__()
        self.emb_model = models.resnet152(pretrained=True) #resnet18, resnet152
        for param in self.emb_model.parameters():
            param.requires_grad = False
        fc_features = self.emb_model.fc.in_features
        self.emb_model.fc = nn.Linear(fc_features, 1024)

        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, num)
        self.params = self.emb_model.fc.parameters()

    def forward(self, x):
        x = self.emb_model(x)
        x = F.relu(x)
        x = self.fc1(x)
        x = F.relu(x)
        return self.fc2(x)

