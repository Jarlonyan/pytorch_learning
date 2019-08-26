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
import conf

my_transform = tv.transforms.Compose([tv.transforms.ToTensor(),  \
                                   tv.transforms.Normalize((0.5,0.5,0.5),(0.56,0.5,0.5))])

my_transform2 = tv.transforms.Compose([tv.transforms.Resize((224, 224)),
                                       tv.transforms.ToTensor(),  \
                                       tv.transforms.Normalize((0.5,0.5,0.5),(0.56,0.5,0.5))])

def train():
    train_set = tv.datasets.CIFAR10(root = "./cifar/",  # MNIST   CIFAR10
                                    download = True,
                                    transform = my_transform)

    train_loader = torch.utils.data.DataLoader(train_set, 
                                               batch_size = conf.batch_size,
                                               shuffle = True,
                                               num_workers = 2)

    #model1
    net = model.Net(num=10)
    params = list(net.parameters())
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    #model2
    #net = model.FinetuneeNet(num=10) #如果用resnet152，就得将图像size设置成为224x224，用my_transform2
    #optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.01)

    #print net

    criterion = nn.CrossEntropyLoss()

    for epoch in range(conf.epochs):
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
            if i%1000 == 999:
                print "epoch=%d, batch=%d, loss=%.4f"%(epoch+1, i, running_loss/100)
                running_loss = 0
                torch.save(net.state_dict(), './checkpoints/dnn_model_%02d_%04d.pkl'%(epoch,i))
    #end-for
    print "end of training"


def test():
    #model1
    net = model.Net(num=10)

    #model2
    #net = model.FinetuneNet(num=10) #如果用resnet152，就得将图像size设置成为224x224，用my_transform2

    net.load_state_dict(torch.load('./checkpoints/dnn_model_00_0420.pkl'))
    batch_size = 20
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
        diff = y_head - labels
        #print y_head, labels, diff
        error_cnt = int(torch.nonzero(diff).shape[0])
        right_rate = (batch_size - error_cnt)*100.0/batch_size
        print right_rate,"%"
        #text = "pred="+str(y_head)+", label="+str(labels)
        #print text #utils.img_show(img, text, color="white")
    #end-for 

def main():
    #train()
    test()


if __name__ == '__main__':
    main()
