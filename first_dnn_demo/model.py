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

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = self.flaten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def flaten(self, x):
        size = x.size()[1:]
        dim = 1
        for s in size:
            dim *= s
        x = x.view(-1, dim)
        return x

def show_a_img(trainset):
    #show image demo
    (img, label) = trainset[2]
    
    show = tv.transforms.ToPILImage()
    img = img.numpy()    # FloatTensor转为ndarray
    img = numpy.transpose(img, (1,2,0)) #把channel那一维放到最后
    plt.imshow(img)
    plt.show()

def main():
    transform = tv.transforms.Compose([tv.transforms.ToTensor(),  \
                                       tv.transforms.Normalize((0.5,0.5,0.5),(0.56,0.5,0.5))])

    trainset = tv.datasets.CIFAR10(root = "./data/",
                                    train = True,
                                    download = False,
                                    transform = transform)

    #show image demo
    #show_a_img(trainset)
    
    trainloader = torch.utils.data.DataLoader(trainset, 
                                              batch_size = 4,
                                              shuffle = True,
                                              num_workers = 2)

    testset = tv.datasets.CIFAR10(root = "./data/",
                                    train = False,
                                    download = False,
                                    transform = transform)

    #constract the net
    net = Net()
    params = list(net.parameters())
    print net

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) 
   
    for epoch in range(0,2):
        running_loss = 0
        for i,data in enumerate(trainloader, 0):
            inputs, labels = data 
            inputs, labels = Variable(inputs), Variable(labels)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.data[0]
            if i%2000==1999:
                print "epoch=%d, batch=%d, loss=%.4f"%(epoch+1, i, running_loss/2000)
                running_loss = 0
    #end-for
    torch.save(net.state_dict(), './model/cnn_params')
    print "end of training"

if __name__ == '__main__':
    main()

