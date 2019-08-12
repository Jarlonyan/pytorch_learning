#coding=utf-8
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.models as models

class VGGNet(nn.Module):
    def __init__(self, dict_size, emb_dim):
        super(VGGNet, self).__init__()
        self.select_layers = ['0', '5', '10', '19', '28']
        self.vgg19 = models.vgg19(pretrained=True).features

    def forward(self, x):
        features = []
        for name,layer in self.vgg19._modules.items():
            x = yaer(x)
            if name in self.select_layers:
                features.append(x)
        return features




















