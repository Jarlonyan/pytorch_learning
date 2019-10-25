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
        self.emb = nn.Embedding(80, 32)
        self.nn_net = nn.Sequential(
            nn.Linear(32, 8),
            #nn.ReLU(),
        )

    def forward(self, x):
        x = self.emb(x)
        x = self.nn_net(x)
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


