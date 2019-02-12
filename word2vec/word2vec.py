#coding=utf-8
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

class SkipGramModel(nn.Module):
    def __init__(self, dict_size, emb_dim):
        super(SkipGramModel, self).__init__()
        self.dict_size = dict_size
        self.emb_dim = emb_dim
        self.in_embeddings = nn.Embedding(dict_size, emb_dim)
        self.out_embeddings = nn.Embedding(dict_size, emb_dim)
        self.init_emb()

    def init_emb(self):
        initrange = 1.0 / self.emb_dim
        self.in_embeddings.weight.data.uniform_(-initrange, initrange)
        self.out_embeddings.weight.data.uniform_(-initrange, initrange)

    def forward(self, pos_in, pos_out, neg_in, neg_out):
        pos_emb_in = self.in_embeddings(pos_in)
        pos_emb_out = self.out_embeddings(pos_out)

        neg_emb_in = self.in_embeddings(neg_in)
        neg_emb_out = self.out_embeddings(neg_out)
        
        pos_prob = torch.mul(pos_emb_in, pos_emb_out)
        pos_prob = torch.sum(pos_prob, 1)
        pos_prob = F.logsigmoid(pos_prob)
        
        neg_prob = torch.mul(neg_emb_in, neg_emb_out)
        neg_prob = torch.sum(neg_prob, 1)
        neg_prob = F.logsigmoid(-1*neg_prob)

        total_prob = torch.sum(pos_prob) + torch.sum(neg_prob)
        return -1*total_prob

    def save_embedding(self, id2word, file_name):
        embeddings = self.in_embeddings.weight.data.numpy()
        with open(file_name, "w") as fout:
            fout.write("word_count=%d, embed_dim=%d \n" % (len(id2word), self.emb_dim))
            for id,w in id2word.items():
                e = embeddings[id]
                e = ' '.join(map(lambda x:str(x),e))
                fout.write("%s = %s \n" % (w, e))
        #end-with