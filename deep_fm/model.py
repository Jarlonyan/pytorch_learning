#coding=utf-8
import torch


class DeepFM(torch.nn.Module):
    def __init__(self, features_sizes, embedding_size,=4, hidden_dims=[32,32], num_classes=1, dropout=[0.5,0.5], verbose=False):
        super().__init__()
        self.field_size = len(feature_sizes)
        self.feature_sizes = feature_sizes
        self.embedding_size = embedding_size
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.dtype = torch
        self.bias = torch.nn.Parameter(torch.rand(1))

        #init fm part
        self.fm_first_order_embeddings = torch.nn.ModuleList(
            [torch.nn.Embedding(feature_size,1) for feature_size in self.feature_sizes]
        )
        self.fm_second_order_embeddings = torch.nn.ModuleList(
            [torch.nn.Embedding(feature_size, self.embedding_size) for feature_size in self.feature_sizes]
        )

        #init deep part
        all_dims = [self.field_size*self.embedding_size] + self.hidden_dims + [self.num_classes]
        for i in range(1, len(hiden_dims)+1):
            setattr(self, 'linear_'+str(i), torch.nn.Linear(all_dims[i-1], all_dims[i])
            setattr(self, 'batchNorm_'+str(i), torch.nn.BatchNorm1d(all_dims[i]))
            setattr(self, 'dropout_'+str(i), torch.nn.Dropout(dropout[i-1]))

    def forward(self, Xi, Xv):
        """
        Xi: a tensor of input's index
        Xv: a tensor of input's value
        """
        fm_first_order_emb_arr = [(torch.sum(emb(Xi[:,i,:], 1).t() * Xv[:,i].t() for i,emb in enumerate(self.fm_first_order_embeddings)))]
        fm_first_order = torch.cat(fm_first_order_emb_arr,1)
        fm_second_order_em_arr = 