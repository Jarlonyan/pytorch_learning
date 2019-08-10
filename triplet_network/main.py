# coding=utf-8
# 参考：https://www.cnblogs.com/king-lps/p/8342452.html
# 数据下载链接：https://files.cnblogs.com/files/king-lps/att_faces.zip

import os
import random
import PIL.ImageOps
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.optim as optim
from torch.autograd import Variable
import torchvision.datasets as dset
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from decimal import Decimal


import conf
import utils
from dataset import MyDataset
import triplet_network

def train():
    train_data = MyDataset(txt=conf.txt_train_data, 
                           transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]),  \
                           should_invert=False)
    train_dataloader = DataLoader(dataset=train_data, \
                                  shuffle=True,       \
                                  batch_size=conf.train_batch_size)
    
    basenet = triplet_network.BaseNet()
    basenet3 = triplet_network.BaseNet3()
    net = triplet_network.TripletNetwork(basenet3)
    criterion = torch.nn.MarginRankingLoss(margin = conf.margin)
    optimizer = optim.Adam(net.parameters(), lr=0.006)

    counter = []
    loss_history = []
    iteration_number = 0

    import matplotlib.pyplot as plt
    plt.ion()
    for epoch in range(0, conf.train_number_epochs):
        for i, data in enumerate(train_dataloader, 0):
            img_a, img_p, img_n= data
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

                plt.plot(counter, loss_history)
                plt.draw()
                plt.xlim((0, 150))
                plt.ylim((0, 5))
                plt.pause(0.03)
    #end-for
    plt.ioff()
    torch.save(net, 'triplet_network.pkl')  # 保存整个神经网络的结构和模型参数 
    plt.show()
    

def test():
    net = torch.load('triplet_network.pkl')
    test_data = MyDataset(txt=conf.txt_test_data, 
                           transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]),  \
                           should_invert=False)
    test_dataloader = DataLoader(dataset=test_data, \
                                  shuffle=True,       \
                                  batch_size=1)
    
    dataiter = iter(test_dataloader)
    for i in range(16):
        img_a, img_p, img_n = next(dataiter)
        concatenated = torch.cat((img_a, img_p, img_n),0)
        
        dist_p, dist_n, embedded_xa, embedded_xp, embedded_xn = net(img_a, img_p, img_n)
    
        dist_p = dist_p.detach().numpy() 
        dist_n = dist_n.detach().numpy()
        text = "dist_p=%.2f"%dist_p+", dist_n=%.2f"%dist_n
        utils.img_show(torchvision.utils.make_grid(concatenated),
                       'Dissimilarity: %s'%text,
                        color="white")

def main():
    train()
    #test()

if __name__ == "__main__":
    main()
