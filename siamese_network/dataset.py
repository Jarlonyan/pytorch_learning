# coding=utf-8
import linecache
import random
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):    # 继承Dataset类以定制
    def __init__(self, txt, transform=None, target_transform=None, should_invert=False):
        self.transform = transform
        self.target_transform = target_transform
        self.should_invert = should_invert
        self.txt = txt       # 之前生成的train.txt

    def __getitem__(self, index):
        line = linecache.getline(self.txt, random.randint(1, self.__len__()))   # 随机选择一个人脸
        line.strip('\n')
        img0_list = line.split()
        
        # 随机数0或1，是否选择同一个人的脸，这里为了保证尽量使匹配和非匹配数据大致平衡（正负类样本相当）
        should_get_same_class = random.randint(0, 1)
        if should_get_same_class:    # 执行的话就挑一张同一个人的脸作为匹配样本对
            while True:
                img1_list = linecache.getline(self.txt, random.randint(1, self.__len__())).strip('\n').split()
                if img0_list[1] == img1_list[1]:
                    break
        else: # else就是随意挑一个人的脸作为非匹配样本对，当然也可能抽到同一个人的脸，概率较小而已
            img1_list = linecache.getline(self.txt, random.randint(1, self.__len__())).strip('\n').split()

        # img_list都是大小为2的列表，list[0]为图像, list[1]为label
        img0 = Image.open(img0_list[0])
        img1 = Image.open(img1_list[0])
        img0 = img0.convert("L")  # 转为灰度
        img1 = img1.convert("L")

        if self.should_invert:   # 是否进行像素反转操作，即0变1,1变0
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:  # 非常方便的transform操作，在实例化时可以进行任意定制
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        # 注意一定要返回数据+标签， 这里返回一对图像+label（应由numpy转为tensor）
        return img0, img1, torch.from_numpy(np.array([int(img1_list[1] == img0_list[1])], dtype=np.float32))

    def __len__(self):       # 数据总长
        fh = open(self.txt, 'r')
        num = len(fh.readlines())
        fh.close()
        return num


