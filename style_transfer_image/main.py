#coding=utf-8
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

import style_transfer_model


'''
实现图像的风格迁移
'''

def main():
    print word2vec_model.in_embeddings.weight.data[0]
    print "end of word2vec"
    word2vec_model.save_embedding(pd.id2word,"data/embedding.txt")

if __name__ == '__main__':
    main()
