# coding=utf-8
import linecache
import random
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class MyDatasetFactory():
    def choose_dataset(self, data_set="cifar", transform=None, download=False):

        if dat_set = 'cifar':
            return  tv.datasets.CIFAR10(root = "./cifar/",
                                            train = True,
                                            download = True,
                                            transform = transform)
        elif data_set == 'MNIST':
            return data_train = tv.datasets.MNIST(root = "./data/",
                                               transform=transform,
                                               train = True,
                                               download = True)
