# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models

import utils

class BaseNet(nn.Module):
    def __init__(self, num=10):
        super(BaseNet, self).__init__()
        self.emb_model = models.resnet152(pretrained=True) #resnet18, resnet152
        for param in self.emb_model.parameters():
            param.requires_grad = False
        fc_features = self.emb_model.fc.in_features
        self.emb_model.fc = nn.Linear(fc_features, 1024)

        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, num)

    def forward(self, x):
        x = self.emb_model(x)
        x = F.relu(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    pass
