#coding=utf-8

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable

# Hyper Parameters
BATCH_SIZE = 64
N_IDEAS = 5             # think of this as number of ideas for generating an art work (Generator)
ART_COMPONENTS = 15     # it could be total point G can draw in the canvas
PAINT_POINTS = np.vstack([np.linspace(-1, 1, ART_COMPONENTS) for _ in range(BATCH_SIZE)])


def artist_works():     # painting from the famous artist (real target)
    a = np.random.uniform(1, 2, size=BATCH_SIZE)[:, np.newaxis]
    paintings = a * np.power(PAINT_POINTS, 2) + (a-1)
    paintings = torch.from_numpy(paintings).float() #numpy.array to torch.FloatTensor
    return paintings

def main():
    #b = artist_works()
    #print b
    #exit(0)

    G = nn.Sequential(nn.Linear(N_IDEAS, 128),
                      nn.ReLU(),
                      nn.Linear(128, ART_COMPONENTS))

    D = nn.Sequential(nn.Linear(ART_COMPONENTS, 128),
                      nn.ReLU(),
                      nn.Linear(128, 1),
                      nn.Sigmoid())
    
    opt_D = torch.optim.Adam(D.parameters(), lr = 0.0001)
    opt_G = torch.optim.Adam(G.parameters(), lr = 0.0001)
    
    plt.ion()

    for step in range(2000):
        x = artist_works()                     # real painting from artist
        z = Variable(torch.randn(BATCH_SIZE, N_IDEAS))  # random ideas
        G_z = G(z)                              # fake painting from G (random ideas) 
        
        prob_artist0 = D(Variable(x))          # D try to increase this prob
        prob_artist1 = D(G_z)                         # D try to reduce this prob
        
        D_loss = - torch.mean(torch.log(prob_artist0) + torch.log(1. - prob_artist1))
        G_loss = torch.mean(torch.log(1. - prob_artist1))

        opt_D.zero_grad()
        D_loss.backward(retain_graph=True)      # reusing computational graph
        opt_D.step()

        opt_G.zero_grad()
        G_loss.backward()
        opt_G.step()

        if step % 50 == 0:  # plotting
            plt.cla()
            plt.plot(PAINT_POINTS[0], G_z.data.numpy()[0], c='#4AD631', lw=3, label='Generated painting',)
            plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')
            plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')
            plt.text(-.5, 2.3, 'D accuracy=%.2f (0.5 for D to converge)' % prob_artist0.data.numpy().mean(), fontdict={'size': 13})
            plt.text(-.5, 2, 'D score= %.2f (-1.38 for G to converge)' % -D_loss.data.numpy(), fontdict={'size': 13})
            plt.ylim((0, 3))
            plt.legend(loc='upper right', fontsize=10)
            plt.draw()
            plt.pause(0.08)

    plt.ioff()
    plt.show()



if __name__ == '__main__':
    main()
