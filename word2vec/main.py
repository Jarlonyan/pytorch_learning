#coding=utf-8
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

from word2vec import SkipGramModel
import dataset

def get_input_layer(word_idx):
    x = torch.zeros(vocabulary_size).float()
    x[word_idx] = 1.0
    return x

def main():
    #idx_pairs, vocab_size = dataset.nlp_get_vocabulary()
    #print idx_pairs
    pd = dataset.ProcessData()
    pd.init_sample_table()
    vocab_size = 60

    word2vec_model = SkipGramModel(vocab_size, 8)
    optimizer = optim.SGD(word2vec_model.parameters(), lr=0.01)

    for epoch in range(10):
        running_loss = 0.0
        batch = 300
        for i in range(batch):
            res = pd.get_batch_pairs(3, 3)
            pos_pairs, neg_pairs = pd.get_pairs_by_neg_sample(res, 5)

            #print "pos=", pos_pairs
            #print "neg=", neg_pairs

            pos_w = [int(pair[0]) for pair in pos_pairs]
            pos_c = [int(pair[1]) for pair in pos_pairs]
            neg_w = [int(pair[0]) for pair in neg_pairs]
            neg_c = [int(pair[0]) for pair in neg_pairs]

            pos_w = Variable(torch.LongTensor(pos_w))
            pos_c = Variable(torch.LongTensor(pos_c))
            neg_w = Variable(torch.LongTensor(neg_w))
            neg_c = Variable(torch.LongTensor(neg_c))

            optimizer.zero_grad()
            loss = word2vec_model(pos_w, pos_c, neg_w, neg_c)
            loss.backward()
            optimizer.step()

            running_loss += loss.data
            if i%80 == 79:
                print "epoch=%d, batch=%d, loss=%.4f"%(epoch+1, i, running_loss)
                running_loss = 0.0
    #end-for
    print word2vec_model.in_embeddings.weight.data[0]
    print "end of word2vec"
    word2vec_model.save_embedding(pd.id2word,"data/embedding.txt")

if __name__ == '__main__':
    main()
