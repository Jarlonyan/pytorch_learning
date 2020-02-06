#coding=utf-8
import torch
import torch.nn.functional as F

class DeepFM(torch.nn.Module):
    def __init__(self, feature_sizes, embedding_size=4, hidden_dims=[32,32], num_classes=1, dropout=[0.5,0.5], verbose=False):
        #super().__init__()
        super(DeepFM, self).__init__()
        self.field_size = len(feature_sizes)
        self.feature_sizes = feature_sizes
        self.embedding_size = embedding_size
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.dtype = torch.long
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
        for i in range(1, len(hidden_dims)+1):
            setattr(self, 'linear_'+str(i), torch.nn.Linear(all_dims[i-1], all_dims[i]))
            setattr(self, 'batchNorm_'+str(i), torch.nn.BatchNorm1d(all_dims[i]))
            setattr(self, 'dropout_'+str(i), torch.nn.Dropout(dropout[i-1]))

    def forward(self, Xi, Xv):
        """
        Xi: a tensor of input's index
        Xv: a tensor of input's value
        """
        fm_first_order_emb_arr = [(torch.sum(emb(Xi[:,i,:], 1).t() * Xv[:,i].t() for i,emb in enumerate(self.fm_first_order_embeddings)))]
        fm_first_order = torch.cat(fm_first_order_emb_arr,1)
        fm_second_order_em_arr = [(torch.sum(emb(Xi[:,i,:]),1).t() * Xv[:,i]).to() for i,emb in enumerate(self.fm_second_order_embeddings)]
        fm_sum_second_order_emb = sum(fm_second_order_em_arr)
        fm_sum_second_order_emb_square = fm_sum_second_order_emb * fm_sum_second_order_emb
        fm_second_order_emb_square_sum = [item*item for item in fm_second_order_emb_arr]
        fm_second_order = (fm_sum_second_order_emb_square - fm_second_order_emb_square_sum) * 0.5

        #deep part
        deep_emb = torch.cat(fm_second_order_emb_arr, 1)
        deep_out = deep_emb
        for i in range(1,len(self.hidden_dims)+1):
            deep_out = getattr(self, 'linear_'+str(i))(deep_out)
            deep_out = getattr(self, 'batchNorm_'+str(i))(deep_out)
            deep_out = getattr(self, "dropout_"+str(i))(deep_out)

        #sum
        total_sum = torch.sum(fm_first_order,1) + torch.sum(fm_second_order,1) + torch.sum(deep_out,1) + self.bias
    
    def train_model(self, loader_train, loader_val, optimizer, epochs=10, verbose=False, print_every=10):
        model = self.train().to(device='cpu')
        criterion = F.binary_cross_entropy_with_logits

        for _ in range(epochs):
            for t,(xi,xv,y) in enumerate(loader_train):
                xi = xi.to(device=torch.device('cpu'), dtype=torch.long)
                xv = xv.to(device=torch.device('cpu'), dtype=torch.float)
                y = y.to(device=torch.device('cpu'), dtype=torch.float)

                total = model(xi, xv)
                loss = criterion(total, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if verbose and t%print_every==0:
                    print ('Iteration %d,loss=%.4f' % (t, loss.item()))
                    print ()

