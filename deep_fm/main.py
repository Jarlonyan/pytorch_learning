#coding=utf-8
#https://github.com/chenxijun1029/DeepFM_with_PyTorch
#Criteo dataset : https://www.kaggle.com/c/criteo-display-ad-challenge

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import sampler
from torch.utils.data import DataLoader

from model import DeepFM
from dataset import CriteoDataset

def main():
    num_train = 900
    train_data = CriteoDataset('./data', train=True)
    loader_train = DataLoader(train_data, batch_size=100, sampler=sampler.SubsetRandomSampler(range(num_train)))

    val_data = CriteoDataset('./data', train=True)
    loader_val = DataLoader(val_data, batch_size=100, sampler=sampler.SubsetRandomSampler(range(num_train, 10000)))

    feature_sizes = np.loadtxt('./data/feature_sizes.txt', delimiter=',')
    feature_sizes = [int(x) for x in feature_sizes]
    print (feature_sizes)

    model = DeepFM(feature_sizes)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0)
    model.train_model(loader_train, loader_val, optimizer, epochs=2, verbose=True)

if __name__ == '__main__':
    main()

