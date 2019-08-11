# coding=utf-8
import linecache
import random
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class MyDataset(Dataset):    # 继承Dataset类以定制
    def __init__(self, txt, transform=None, target_transform=None, should_invert=False):
        self.transform = transform
        self.target_transform = target_transform
        self.should_invert = should_invert
        self.txt = txt       # 之前生成的train.txt

    def __getitem__(self, index):
        line = linecache.getline(self.txt, random.randint(1, self.__len__()))   # 随机选择一个人脸
        line.strip('\n')
        img_a_list = line.split()
        
        while True:
            img_p_list = linecache.getline(self.txt, random.randint(1, self.__len__())).strip('\n').split()
            if img_a_list[1] == img_p_list[1]:
                break

        while True:
            img_n_list = linecache.getline(self.txt, random.randint(1, self.__len__())).strip('\n').split()
            if img_a_list[1] != img_n_list[1]:
                break

        # img_list都是大小为2的列表，list[0]为图像, list[1]为label
        img_a = Image.open(img_a_list[0])
        img_p = Image.open(img_p_list[0])
        img_n = Image.open(img_n_list[0])
        img_a = img_a.convert("RGB")  # 转为灰度是L, 转成3通道是RGB
        img_p = img_p.convert("RGB")
        img_n = img_n.convert("RGB")
        
        if self.transform is not None:  # 非常方便的transform操作，在实例化时可以进行任意定制
            img_a = self.transform(img_a)
            img_p = self.transform(img_p)
            img_n = self.transform(img_n)

        return img_a, img_p, img_n

    def __len__(self):       # 数据总长
        fh = open(self.txt, 'r')
        num = len(fh.readlines())
        fh.close()
        return num


