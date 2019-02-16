#coding=utf-8
#参考：https://github.com/chenyuntc/pytorch-book/blob/master/chapter6-%E5%AE%9E%E6%88%98%E6%8C%87%E5%8D%97/models/alexnet.py

import torch as t
from torch import nn
from .basic_module import BasicModule

class AlexNet(BasicModule):
    def __init__(self, num_classes=2):
        super(Alex, self).__init__()
        self.model_name = 'AlexNet'

        self.feature_net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.classifier_net = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256*6*6, 4096)
        )

    def forward(self, x):
        x = self.feature_net(x)
        x = x.view(x.size(0), 256*6*6)
        x = self.classifier_net(x)
        return x