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

class FinetuneNet(nn.Module):
    def __init__(self, num_classes=10):
        super(FinetuneNet, self).__init__()
        self.emb_model = models.resnet152(pretrained=True) #resnet18, resnet152
        for param in self.emb_model.parameters():
            param.requires_grad = False
        fc_size = self.emb_model.fc.in_features
        self.emb_model.fc = nn.Linear(fc_size, 1024)
        self.nn_net = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        #import pdb; pdb.set_trace()
        x = self.emb_model(x)
        x = self.nn_net(x)
        return x

