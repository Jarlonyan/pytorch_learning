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
        img_list = line.split()
        
        # img_list都是大小为2的列表，list[0]为图像, list[1]为label
        img = Image.open(img_list[0])
        label = int(img_list[1])
        #img = img.convert("RGB")  # 转为灰度是L, 转成3通道是RGB
        if self.transform is not None:  # 非常方便的transform操作，在实例化时可以进行任意定制
            img = self.transform(img)

        return img, label # torch.from_numpy(np.array([label], dtype=np.long))

    def __len__(self):       # 数据总长
        fh = open(self.txt, 'r')
        num = len(fh.readlines())
        fh.close()
        return num


