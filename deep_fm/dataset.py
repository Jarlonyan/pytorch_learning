#coding=utf-8

import torch
from torch.utils.data import Dataset

import pandas as pd
import numpy as np
import os

continous_features = 13

class CriteoDataset(torch.utils.data.Dataset):
    def __init__(self, root, train=True):
        self.root = root
        self.train = train

        if not os.path.exists(self.root):
            raise RuntimeError("dataset is not found.")

        if self.train:
            data = pd.read_csv(os.path.join(root, 'train.txt'))
            self.train_data = data.iloc[:, :-1].values
            self.target = data.iloc[:, -1].values
        else:
            data = pd.read_csv(os.path.join(root, 'test.txt'))
            self.test_data = data.iloc[:, -1].values

    def __getitem__(self, idx):
        data_i, target_i = self.train_data[idx, :], self.target[idx]
        Xi_continous = np.zeros_like(data_i[: 13])
        Xi_categorial = data_i[13:]
        Xi = torch.from_numpy(np.concatenate((Xi_continous, Xi_categorial)).astype(np.int32)).unsqueeze(-1)

        Xv_categorial = np.ones_like(data_i[continous_features:])
        Xv_continous = data_i[:continous_features]
        Xv = torch.from_numpy(np.concatenate((Xv_continous,  Xv_categorial)).astype(np.int32))
        return Xi, Xv, target_i

    def __len__(self):
        return len(self.train_data)


