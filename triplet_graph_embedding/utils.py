# coding=utf-8
import os
import networkx as nx
import numpy as np
import random
import pickle

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

def generate_ins():
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

def generate_word_dict():
    G = nx.read_gml("data/dolphins.gml")
    word2idx = dict()
    idx = 0
    for i in G.nodes:
        word2idx[i] = idx
        idx += 1

def main():
    #generate_word_dict()
    generate_ins()

if __name__ == "__main__":
    main()

