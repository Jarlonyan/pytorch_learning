# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models

import utils

class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet3, self).__init__()
        self.model = models.resnet152(pretrained=True) #resnet18, resnet152
        for param in self.model.parameters():
            param.requires_grad = False
        fc_features = self.model.fc.in_features
        self.model.fc = nn.Linear(fc_features, 128)

        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)

    def forward(self, x):
        x = self.model(x)
        x = F.relu(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    pass
