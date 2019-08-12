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
import model

def train():
    train_data = MyDataset(txt=conf.txt_train_data, 
                           transform=transforms.Compose([transforms.Resize((224, 224)), \
                                                         transforms.ToTensor(),  \
                                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])]),  \
                           should_invert=False)
    train_dataloader = DataLoader(dataset=train_data, \
                                  shuffle=True,       \
                                  batch_size=conf.train_batch_size)
    
    net = model.BaseNet()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.006)

    counter = []
    loss_history = []
    iteration_number = 0

    plt.ion()
    for epoch in range(0, conf.train_number_epochs):
        for i, data in enumerate(train_dataloader, 0):
            img, y = data
            img = Variable(img)
            y_head = net(img)

            loss = criterion(y_head, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if True:   #i % 1 == 0:
                print "Epoch={}, i={}, current loss={}".format(epoch, i, loss.data)
                iteration_number += 1
                counter.append(iteration_number)
                loss_history.append(loss.data)

                plt.plot(counter, loss_history)
                plt.draw()
                plt.xlim((0, 100))
                plt.ylim((0, 4))
                plt.pause(0.03)
    #end-for
    plt.ioff()
    torch.save(net, 'hand_classifier_model.pkl')  # 保存整个神经网络的结构和模型参数 
    plt.show()
    

def test():
    net = torch.load('hand_classifier_model.pkl')
    test_data = MyDataset(txt=conf.txt_test_data, 
                           transform=transforms.Compose([transforms.Resize((224, 224)),  \
                                                        transforms.ToTensor(),            \
                                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])]),  \
                           should_invert=False)
    test_dataloader = DataLoader(dataset=test_data, \
                                  shuffle=True,       \
                                  batch_size=1)
    
    dataiter = iter(test_dataloader)
    for i in range(16):
        img, label = next(dataiter)
        y_head = net(img)
        y_head = y_head[0].tolist()

        #print y_head
        idx = y_head.index(max(y_head))
        text = "output="+str(idx)+", label="+str(int(label))
        print text #utils.img_show(img, text, color="white")
    #end-for

def main():
    train()
    #test()

if __name__ == "__main__":
    main()
