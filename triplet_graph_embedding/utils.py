# coding=utf-8
import os
import networkx as nx
import numpy as np
import random
import pickle
from collections import defaultdict

import conf

def rand_select_neg(G, target):
    nodes_list = list(G.nodes)
    idx = 0
    while True:
        idx = random.randint(0, len(nodes_list)-1)
        neg = nodes_list[idx]
        if neg not in G.neighbors(target):
            break
    return nodes_list[idx]

def generate_ins_1order():
    G = nx.read_gml("data/dolphins.gml", label='id')
    f = open(conf.data_path, 'w')
    word2idx = dict()
    idx2word = dict()
    idx = 0
    for i in G.nodes:
        word2idx[i] = idx
        idx2word[idx] = i
        idx += 1

    for i in G.nodes:
        node_a = i
        for j in G.neighbors(i):
            node_p = j 
            node_n = rand_select_neg(G, i)
            if node_a in word2idx.keys():
                a = word2idx[node_a]
                p = word2idx[node_p]
                n = word2idx[node_n]
                f.write(str(a)+'\t'+str(p)+'\t'+str(n)+'\n')
            else:
                print "error, ", node_a
        #end-for
    #end-for
    f.close()

    with open('data/idx2word.pickle', 'wb') as f:
        pickle.dump(idx2word, f)
    
    with open('data/word2idx.pickle', 'wb') as f:
        pickle.dump(word2idx, f)

def rand_select_neg2(G, target, neighbors):
    nodes_list = list(G.nodes)
    idx = 0
    while True:
        idx = random.randint(0, len(nodes_list)-1)
        neg = nodes_list[idx]
        if neg not in neighbors:
            break
    return neg

def rand_select_pos2(G, target, neighbors):
    nodes_list = list(G.nodes)
    idx = 0
    while True:
        idx = random.randint(0, len(nodes_list)-1)
        pos = nodes_list[idx]
        if pos in neighbors:
            break
    return pos

def generate_ins_xorder():
    G = nx.read_gml("data/dolphins.gml", label='id')
    f = open(conf.data_path, 'w')
    
    two_order = defaultdict(set)

    word2idx = dict()
    idx2word = dict()
    idx = 0
    for i in G.nodes:
        word2idx[i] = idx
        idx2word[idx] = i
        idx += 1

    for i in G.nodes:
        for j in G.neighbors(i):
            for k in G.neighbors(j):
                if k != i:
                    two_order[i].add(k)
            #end-for
        #end-for
    #end-for

    for i in two_order.keys():
        one_order_nei = set(G.neighbors(i))
        two_order_nei = two_order[i]

        for k in range(len(one_order_nei)):
            a = word2idx[i]
            p = word2idx[rand_select_pos2(G, i, one_order_nei.union(two_order_nei))]
            n = word2idx[rand_select_neg2(G, i, one_order_nei.union(two_order_nei))]

            f.write(str(a)+'\t'+str(p)+'\t'+str(n)+'\n')
    f.close()
    
    with open('data/idx2word.pickle', 'wb') as f:
        pickle.dump(idx2word, f)
    
    with open('data/word2idx.pickle', 'wb') as f:
        pickle.dump(word2idx, f)

def main():
    generate_ins_1order()
    #generate_ins_xorder()

if __name__ == "__main__":
    main()

