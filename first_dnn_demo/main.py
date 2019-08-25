#coding=utf-8

import numpy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision as tv
from torchvision.transforms import ToPILImage
from torch.autograd import Variable
import matplotlib.pyplot as plt

import model
import utils

my_transform = tv.transforms.Compose([tv.transforms.ToTensor(),  \
                                   tv.transforms.Normalize((0.5,0.5,0.5),(0.56,0.5,0.5))])

def train():
    train_set = tv.datasets.CIFAR10(root = "./cifar/",  # MNIST   CIFAR10
                                    download = True,
                                    transform = my_transform)

    train_loader = torch.utils.data.DataLoader(train_set, 
                                               batch_size = 4,
                                               shuffle = True,
                                               num_workers = 2)

    #constract the net
    net = model.Net(num=10)
    params = list(net.parameters())
    print net

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) 
   
    for epoch in range(0,2):
        running_loss = 0
        for i,data in enumerate(train_loader, 0):
            inputs, labels = data 
            inputs, labels = Variable(inputs), Variable(labels)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.data
            if i%2000 == 1999:
                print "epoch=%d, batch=%d, loss=%.4f"%(epoch+1, i, running_loss/2000)
                running_loss = 0
    #end-for
    torch.save(net.state_dict(), './checkpoints/model.pkl')
    print "end of training"


def test():
    batch_size = 2
    net = model.Net(num=10)
    net.load_state_dict(torch.load('./checkpoints/model.pkl'))
    test_data = tv.datasets.CIFAR10(root = "./cifar/",
                                   train = False,
                                   download = True,
                                   transform = my_transform)

    test_dataloader = torch.utils.data.DataLoader(dataset=test_data,  \
                                                  shuffle=True,       \
                                                  batch_size=batch_size)

    dataiter = iter(test_dataloader)
    for i in range(10):
        imgs, labels = next(dataiter)
        y_head = net(imgs)
        y_head = y_head.data.max(1, keepdim=True)[1].view(batch_size)
        #text = "y_head="+str(y_head[0].tolist())+", pred="+str(idx)+", label="+str(int(labels))
        diff = y_head - labels
        text = "pred="+str(y_head)+", label="+str(labels)
        print text #utils.img_show(img, text, color="white")
    #end-for 

def main():
    train()
    #test()


if __name__ == '__main__':
    main()
