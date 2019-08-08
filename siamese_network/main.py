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

import conf
import utils
from dataset import MyDataset
import siamese_network

def train():
    #utils.convert()
    #exit(0)
    train_data = MyDataset(txt=conf.txt_train_data, 
                           transform=transforms.Compose([transforms.Resize((100, 100)), transforms.ToTensor()]),  \
                           should_invert=False)
    train_dataloader = DataLoader(dataset=train_data, \
                                  shuffle=True,       \
                                  batch_size=conf.train_batch_size)
    net = siamese_network.SiameseNetwork()
    criterion = siamese_network.ContrastiveLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0006)

    counter = []
    loss_history = []
    iteration_number = 0

    import matplotlib.pyplot as plt
    plt.ion()
    for epoch in range(0, conf.train_number_epochs):
        for i, data in enumerate(train_dataloader, 0):
            img1, img2, label = data
            img1, img2, label = Variable(img1), Variable(img2), Variable(label)
            output1, output2 = net(img1, img2)
            
            loss_contrastive = criterion(output1, output2, label)
            loss_contrastive.backward()
            optimizer.step()

            if i % 2 == 0:
                print "Epoch{}, current loss={}".format(epoch, loss_contrastive.data[0])
                iteration_number += 1
                counter.append(iteration_number)
                loss_history.append(loss_contrastive.data[0])

                print counter, '===>', loss_history
                plt.plot(counter, loss_history)
                plt.draw()
                plt.xlim((0, 50))
                plt.ylim((0, 20))
                plt.pause(0.08)
    #utils.show_plot(counter, loss_history)
    plt.ioff()
    plt.show()
    torch.save(net, 'siameses_network.pkl')  # 保存整个神经网络的结构和模型参数 

def test():
    model_path ='siameses_network.pkl'
    model = torch.load(model_path)

    train_data = MyDataset(txt=conf.txt_train_data, 
                           transform=transforms.Compose([transforms.Resize((100, 100)), transforms.ToTensor()]),  \
                           should_invert=False)
    train_dataloader = DataLoader(dataset=train_data, \
                                  shuffle=True,       \
                                  batch_size=conf.train_batch_size)
    net = siamese_network.SiameseNetwork()
    criterion = siamese_network.ContrastiveLoss()

    data_iter = iter(train_dataloader)
    img1, img2, label = next(data_iter)
    output1, output2 = net(img1, img2)
    #loss_contrastive = criterion(output1, output2, label)
    print output1
    #print loss_contrastive

    '''img1, img2, label = data
    img1, img2, label = Variable(img1), Variable(img2), Variable(label)
    output1, output2 = net(img1, img2)
    loss_contrastive = criterion(output1, output2, label)
    '''

def main():
    train()
    #test()

if __name__ == "__main__":
    main()
