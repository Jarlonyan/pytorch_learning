# coding=utf-8
# 参考：https://www.cnblogs.com/king-lps/p/8342452.html
# 数据下载链接：https://files.cnblogs.com/files/king-lps/att_faces.zip

import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.autograd import Variable
import torchvision.datasets as dset
import torch.nn.functional as F
import numpy as np
from decimal import Decimal


import conf
import utils
from dataset import MyDataset
import model

def train():
    train_data = MyDataset(txt=conf.data_path) 
    train_dataloader = DataLoader(dataset=train_data, \
                                  shuffle=True,       \
                                  batch_size=conf.train_batch_size)
    
    basenet = model.BaseNet()
    net = model.TripletNetwork(basenet)
    criterion = torch.nn.MarginRankingLoss(margin = conf.margin)
    optimizer = optim.Adam(net.parameters(), lr=0.006)

    counter = []
    loss_history = []
    iteration_number = 0

    for epoch in range(0, conf.train_number_epochs):
        for i, data in enumerate(train_dataloader, 0):
            img_a, img_p, img_n = data
            img_a, img_p, img_n = Variable(img_a), Variable(img_p), Variable(img_n)
            dist_p, dist_n, embedded_xa, embedded_xp, embedded_xn = net(img_a, img_p, img_n)
            target = torch.FloatTensor(dist_p.size()).fill_(1)
            target = Variable(target)
            loss_triplet = criterion(dist_n, dist_p, target)
            loss_embed = embedded_xa.norm(2) + embedded_xp.norm(2) + embedded_xn.norm(2)
            loss = loss_triplet + 0.001*loss_embed

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 3 == 0:
                print "Epoch{}, current loss={}".format(epoch, loss.data)
                iteration_number += 1
                counter.append(iteration_number)
                loss_history.append(loss.data)
                torch.save(net.state_dict(), './checkpoints/model_%01d_%03d.pkl'%(epoch, i))
    #end-for

def test():
    basenet = model.BaseNet()
    net = model.TripletNetwork(basenet)
    net.load_state_dict(torch.load('./checkpoints/model_4_030.pkl'))
    
    test_data = MyDataset(txt=conf.data_path)
    test_dataloader = DataLoader(dataset=test_data, \
                                  shuffle=True,       \
                                  batch_size=1)
    
    dataiter = iter(test_dataloader)
    for i in range(1):
        ins_a, ins_p, ins_n = next(dataiter)
        dist_p, dist_n, embedded_xa, embedded_xp, embedded_xn = net(ins_a, ins_p, ins_n)
    
        dist_p = dist_p.detach().numpy() 
        dist_n = dist_n.detach().numpy()
        print dist_p

def main():
    #train()
    test()

if __name__ == "__main__":
    main()
