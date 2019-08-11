# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models

import utils

class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),
            nn.Dropout2d(p=0.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.Dropout2d(p=0.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.Dropout(p=0.2)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(8*100*100, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 32)
        )

    def forward(self, x):
        output = self.cnn1(x)
        output = output.view(x.size()[0], -1)
        output = self.fc1(output)
        return output
    
class BaseNet2(nn.Module):
    def __init__(self):
        super(BaseNet2, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(9680, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return self.fc2(x)

class BaseNet3(nn.Module):
    def __init__(self):
        super(BaseNet3, self ).__init__() 
        self.model = models.resnet152(pretrained=True) #resnet18, resnet152
        for param in self.model.parameters():
            param.requires_grad = False
        fc_features = self.model.fc.in_features
        self.model.fc = nn.Linear(fc_features, 64)

        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)

    def forward(self, x):
        x = self.model(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class TripletNetwork(nn.Module):
    def __init__(self, embeddingnet):
        super(TripletNetwork, self).__init__()
        self.embeddingnet = embeddingnet

    def forward(self, xa, xp, xn):
        embedded_xa = self.embeddingnet(xa)
        embedded_xp = self.embeddingnet(xp)
        embedded_xn = self.embeddingnet(xn)
        dist_p = F.pairwise_distance(embedded_xa, embedded_xp, 2)
        dist_n = F.pairwise_distance(embedded_xa, embedded_xn, 2)
        return dist_p, dist_n, embedded_xa, embedded_xp, embedded_xn


if __name__ == "__main__":
    pass
