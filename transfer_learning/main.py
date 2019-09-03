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


my_transforms = {
    'train': tv.transforms.Compose([tv.transforms.ToTensor(),  \
                                   tv.transforms.Normalize((0.5,0.5,0.5),(0.56,0.5,0.5))
    ]),
    'test': tv.transforms.Compose([tv.transforms.ToTensor(),  \
                                   tv.transforms.Normalize((0.5,0.5,0.5),(0.56,0.5,0.5))
    ])
}

def train():
    train_set = tv.datasets.CIFAR10(root = "./cifar/",  # MNIST   CIFAR10
                                    download = True,
                                    transform = my_transforms['train'])

    train_loader = torch.utils.data.DataLoader(train_set, 
                                               batch_size = conf.batch_size,
                                               shuffle = True,
                                               num_workers = 2)

    # finetune model
    net = model.FinetuneNet(num_classes=10) #如果用resnet152，就得将图像size设置成为224x224，用my_transform2
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(net.parameters())

    for epoch in range(conf.epochs):
        running_loss = 0
        for i,data in enumerate(train_loader, 0):
            inputs, labels = data 
            inputs, labels = Variable(inputs), Variable(labels)
            outputs = net(inputs)

            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.data
            if i%100 == 99:
                print "epoch=%d, batch=%d, loss=%.4f"%(epoch+1, i, running_loss/100)
                running_loss = 0
                torch.save(net.state_dict(), './checkpoints/dnn_model_%02d_%04d.pkl'%(epoch,i))
    #end-for
    print "end of training"


def test():
    #model
    net = model.FinetuneNet(num_classes=10) #如果用resnet152，就得将图像size设置成为224x224，用my_transform2
    net.load_state_dict(torch.load('./checkpoints/dnn_model_01_0299.pkl'))
    batch_size = 20
    test_data = tv.datasets.CIFAR10(root = "./cifar/",
                                   train = False,
                                   download = True,
                                   transform = my_transforms['test'])

    test_dataloader = torch.utils.data.DataLoader(dataset=test_data,  \
                                                  shuffle=True,       \
                                                  batch_size=batch_size)

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


if __name__ == '__main__':
    main()
