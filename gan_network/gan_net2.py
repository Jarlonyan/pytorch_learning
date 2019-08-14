#coding=utf-8

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable

batch_size = 4

#generate real data
def gen_real_data():
    X = np.random.normal(size=(1000, 2))
    A = np.array([[1,2], [-0.1, 0.5]])
    b = np.array([1,2])
    X = np.dot(X, A) + b
    return X

def iter_minibatch(x, batch_size, shuffle=True):
    indices = np.arange(x.shape[0])
    if shuffle:
        np.random.shuffle(indices)
    for i in range(0, x.shape[0], batch_size):
        yield x[indices[i:i+batch_size], :]


class GAN(object):
    def __init__(self):
        self.netG = nn.Linear(2, 2)
        self.netD = nn.Sequential(nn.Linear(2, 5), nn.Tanh(),
                                  nn.Linear(5, 3), nn.Tanh(),
                                  nn.Linear(3, 2))
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, z):
        self.z = Variable((torch.from_numpy(z)).float())
        self.fake_x = self.netG(self.z)
        self.real_x = Variable((torch.from_numpy(x)).float())
        self.label = Variable(torch.LongTensor(x.shape[0]).fill_(1), requires_grad=False)

    def backward_D(self):
        pred_fake = self.netD(self.fake_x.detach())
        self.loss_D_fake = self.criterion(pred_fake, self.label*0)
        pred_real = self.netD(self.real_x)
        self.loss_D_real = self.criterion(pred_real, self.label*1)
        self.loss_D = self.loss_D_fake + self.loss_D_real
        self.loss_D.backward()

    def backward_G(self):
        pred_fake = self.netD(self.fake_x)
        self.loss_G = self.criterion(pred_fake, self.label*1) #fool the D
        self.loss_G.backward()

    def test(self, x, z):
        z = Variable((torch.from_numpy(z)).float(), volatile=True)
        fake_x = self.netG(z)
        real_x = Variable((torch.from_numpy(x)).float(), volatile=True)

        pred_fake = self.netD(fake_x)
        pred_real = self.netG(real_x)
        return fake_x.data.numpy(), pred_real.data.numpy(), pred_fake.data.numpy()



def main():
    X = gen_real_data()

    gan = GAN()
    d_optim = torch.optim.Adam(gan.netD.parameters(), lr=0.05)
    g_optim = torch.optim.Adam(gan.netG.parameters(), lr=0.01)

    #init weight in the model
    def init_weights(m):
        if type(m) == nn.Linear:
            m.weight.data.normal_(0.0, 0.02)

    gan.netD.apply(init_weights)
    gan.netG.apply(init_weights)


    for epoch in range(10):
        avg_loss = 0
        count = 0

        for x_batch in iter_minibatch(X, batch_size):
            #generate noize z
            z_batch = np.random.normal(size=(4,2))

            #forward
            gan.forward(x_batch, z_batch)

            # update D network
            d_optim.zero_grad()
            gan.backward_D()
            d_optim.step()
        
            # update G network
            g_optim.zero_grad()
            gan.backward_G()
            g_optim.step()

            avg_loss += gan.loss_D.data.numpy()
            count += 1
        #end-for
        avg_loss /= count
        z = np.random.normal(size=(100,2))
        experpt = np.random.randint(1000, size=100)
        fake_x, pred_real, pred_fake = gan.test(X[experpt,:], z)
        accuracy = 0.5*(np.sum(pred_real[:,1]>pred_real[:,0])/100. + np.sum(pred_fake[:,0]>pred_fake[:,1])/100.)

        print "D loss at epoch %d: %f" % (epoch, avg_loss)
        print "D accuracy at epoch %d: %f" %(epoch, accuracy)
    #end-for

if __name__ == '__main__':
    main()


