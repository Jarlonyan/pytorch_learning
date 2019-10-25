# coding=utf-8
#https://github.com/andreasveit/triplet-network-pytorch/blob/master/tripletnet.py

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
import pickle
import networkx as nx

import conf
import utils
from dataset import MyDataset
import model

def train():
    train_data = MyDataset(txt=conf.data_path) 
    train_dataloader = DataLoader(dataset=train_data, shuffle=True, batch_size=conf.train_batch_size)
    
    basenet = model.BaseNet()
    net = model.TripletNetwork(basenet)
    criterion = torch.nn.MarginRankingLoss(margin = conf.margin)
    optimizer = optim.Adam(net.parameters(), lr=0.01)

    counter = []
    loss_history = []
    iteration_number = 0

    for epoch in range(0, conf.train_number_epochs):
        for i, data in enumerate(train_dataloader, 0):
            ins_a, ins_p, ins_n = data
            ins_a, ins_p, ins_n = Variable(ins_a), Variable(ins_p), Variable(ins_n)
            dist_p, dist_n, embedded_xa, embedded_xp, embedded_xn = net(ins_a, ins_p, ins_n)
            #import pdb; pdb.set_trace()
            #print "dist=",dist_p.size()
            target = torch.FloatTensor(dist_p.size()).fill_(1)
            target = Variable(target)
            loss_triplet = criterion(dist_n, dist_p, target) #loss_triplet=max(0, dp-dn+margin)
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
    #end-for

def test():
    basenet = model.BaseNet()
    net = model.TripletNetwork(basenet)
    net.load_state_dict(torch.load('./checkpoints/model_14_030.pkl'))
    
    test_data = MyDataset(txt=conf.data_path)
    test_dataloader = DataLoader(dataset=test_data, shuffle=True, batch_size=1)
    
    dataiter = iter(test_dataloader)
    for i in range(1):
        ins_a, ins_p, ins_n = next(dataiter)
        dist_p, dist_n, embedded_xa, embedded_xp, embedded_xn = net(ins_a, ins_p, ins_n)
        print ins_a,ins_p,ins_n
        dist_p = dist_p.detach().numpy() 
        dist_n = dist_n.detach().numpy()
        print dist_p

def get_embedding():
    basenet = model.BaseNet()
    net = model.TripletNetwork(basenet)
    #net.load_state_dict(torch.load('./checkpoints/model_14_030.pkl'))
    net.load_state_dict(torch.load('./checkpoints/model_0_000.pkl'))

    G = nx.read_gml("data/dolphins.gml")
    with open('data/word2idx.pickle', 'rb') as f:
        word2idx = pickle.load(f)
    for i in G.nodes():
        idx = word2idx[i]
        ins = torch.LongTensor([idx])
        for j in G.neighbors(i):
            p_idx = word2idx[j]
            p_ins = torch.LongTensor([p_idx])
            
            n = utils.rand_select_neg(G, i)
            n_idx = word2idx[n]
            n_ins = torch.LongTensor([n_idx])

            dist_1, dist_2, embedded_x1, embedded_x2, embedded_x3 = net(ins, p_ins, n_ins)
            print dist_1.detach().numpy().tolist()[0], dist_2.detach().numpy().tolist()[0]
        #print i, '\t', list(embedded_x1.detach().numpy().tolist())[0]
        

def main():
    #train()
    #test()
    get_embedding()

if __name__ == "__main__":
    main()
