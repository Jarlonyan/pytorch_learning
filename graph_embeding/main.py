#coding=utf-8
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import networkx as nx
import matplotlib.pylab as plt

def main():
    #G = nx.Graph()
    G = nx.tutte_graph() #petersen_graph()
    print G.edges()

    #draw G
    nx.draw(G)
    plt.show()

if __name__ == '__main__':
    main()
