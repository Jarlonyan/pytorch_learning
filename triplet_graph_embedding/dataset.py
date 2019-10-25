# coding=utf-8
import linecache
import random
import numpy as np
import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):    # 继承Dataset类以定制
    def __init__(self, txt):
        self.txt = txt

    def __getitem__(self, index):
        line = linecache.getline(self.txt, random.randint(1, self.__len__()))   # 随机选择一个人脸
        line.strip('\n')
        ins_list = line.split('\t')
        
        ins_a = torch.LongTensor([int(ins_list[0])])
        ins_p = torch.LongTensor([int(ins_list[1])])
        ins_n = torch.LongTensor([int(ins_list[2])])
        
        return ins_a, ins_p, ins_n

    def __len__(self):       # 数据总长
        fh = open(self.txt, 'r')
        num = len(fh.readlines())
        fh.close()
        return num


