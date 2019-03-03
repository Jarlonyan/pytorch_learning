#coding=utf-8
# 参考：https://github.com/andreasveit/triplet-network-pytorch

from __future__ import print_function
import argparse
import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd imprt Variable
import torch.backends.cudnn as cudnn
from visdom import Visdom
import numpy as np
from triplet_network import Net, Tripletnet

#training settings
parser = argparse.ArgumentParser(description='pytorch MNIST example')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='input batch size for training(default:64)')
parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N',
                    help='input batch size for testing(default:1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train(default:10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate(default:0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum(default:0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed(default:1)')
parser.add_argument('--log_interval', type=int, default=20, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--margin', type=float, default=0.2, metavar='M',
                    help='margin for triplet loss(default:0.2)')
parser.add_argument('--resume', type=str, default='', 
                    help='path to latest checkpoint(default:none)')
parser.add_argument('--name', type=str, default='TripletNet', 
                    help='name of experiment')

best_acc = 0

def main():
    global args, best_acc
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    global plotter
    plotter = VisdomLinePlotter(env_name=args.name)
    
    kwargs = {'num_workers':1, 'pin_memory':True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader()

    model = Net()
    tnet = Tripletnet(model)
    if args.cuda:
        tnet.cuda()
    if args.resume:
        if os.path.isfile(args.resume):
            print ("----> loading checkpoint '{}'".format(args.resume))
        else:
            print ("----> no checkpoint found at '{}'".format(args.resume))
    cudnn.benchmark = True
    criterion = torch.nn.MarginRankingLoss(margin=args.margin)
    optimizer = optim.SGD(tnet.parameters(), lr=args.lr, momentum=args.momentum)

if __name__ == "__main__":
    main()