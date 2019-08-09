# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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
