# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import utils

class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=6)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return self.fc2(x)

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

# 定制化的对比loss
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label)*torch.pow(euclidean_distance, 2) +
                                      label * torch.pow(torch.clamp(self.margin-euclidean_distance, min=0.0), 2))
        return loss_contrastive


if __name__ == "__main__":
    pass
