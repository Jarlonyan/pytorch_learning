# coding=utf-8

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
                                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                           ]),
                           should_invert=False
    )

    train_dataloader = DataLoader(dataset=train_data, \
                                  shuffle=True,       \
                                  batch_size=conf.batch_size
    )
    
    net = model.FinetuneNet(num_classes=6)
    criterion = torch.nn.NLLLoss()
    optimizer = optim.Adam(net.parameters())
    #optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.01)

    counter = []
    loss_history = []
    iteration_number = 0

    #plt.ion()
    for epoch in range(0, conf.epochs):
        for i, data in enumerate(train_dataloader, 0):
            img, y = data
            img = Variable(img)
            y_heads = net(img)

            optimizer.zero_grad()
            loss = criterion(y_heads, y)
            loss.backward()
            optimizer.step()

            if  i % 2 == 0:
                print "Epoch={}, i={}, current loss={}".format(epoch, i, loss.data)
                iteration_number += 1
                counter.append(iteration_number)
                loss_history.append(loss.data)
                print loss.data
                torch.save(net.state_dict(), './checkpoints/hand_classifier_model_%01d_%03d.pkl'%(epoch, i))  # 保存整个神经网络的结构和模型参数

                '''
                plt.plot(counter, loss_history)
                plt.draw()
                plt.xlim((0, 100))
                plt.ylim((0, 30))
                plt.pause(0.03)
                '''
    #end-for
    #plt.ioff()
    #plt.show()
    

def test():
    net = model.FinetuneNet(num_classes=6)
    net.load_state_dict(torch.load('./checkpoints/hand_classifier_model_0_008.pkl'))
    test_data = MyDataset(txt=conf.txt_test_data, 
                          transform=transforms.Compose([transforms.Resize((224, 224)),  \
                                                        transforms.ToTensor(),            \
                                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                          ]),
                          should_invert=False
    )

    test_dataloader = DataLoader(dataset=test_data, \
                                  shuffle=True,       \
                                  batch_size=conf.batch_size)
    
    dataiter = iter(test_dataloader)
    for i in range(6):
        imgs, labels = next(dataiter)
        log_prob = net(imgs)
        prob = torch.exp(log_prob)
        # 找到概率最大的
        pred = torch.max(prob, dim=1)

        # 计算accuracy
        equals = (pred.indices == labels).float()
        accuracy = torch.mean(equals)
        print float(accuracy)*100,'%' #prob, labels
    #end-for


def main():
    #train()
    test()

if __name__ == "__main__":
    main()
