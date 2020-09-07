#coding=utf-8
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

#问 Self Attention，并实现 self_attention_layer
# (B, S, H)
class SelfAttention(nn.Module):
    def __init__(self):
        pass
    def forward(self, input_seq, lens):
        batch_size, max_seq_len, hidden_dim = input_seq.size()
        input_seq_trans = transpose(input_seq, dim=-1) # B, H, S
        score = torch.matmul(input_seq, input_seq_trans) # B, S, S
        score = softmax(score, dim=1) # B, S, S
        score = reduce_sum(score) # B, S, 1
        weights = mul(score, input_seq) # brocast multiply
        content = reduce_sum(weights, dim=-1)
        return content # B, S, H

#视觉应用中的self-attention机制
class Self_Attn_Spatial(nn.Module):
    """ 
    func: Self attention Spatial Layer 自注意力机制.通过类似Transformer中的Q K V来实现
    inputs:
        in_dim: 输入的通道数
        out_dim: 在进行self attention时生成Q,K矩阵的列数, 一般默认为in_dim//8
    """
    def __init__(self,in_dim,out_dim):
        super(Self_Attn_Spatial,self).__init__()
        self.chanel_in = in_dim
        self.out_dim = out_dim
 
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = out_dim , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = out_dim , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))
 
        self.softmax  = nn.Softmax(dim=-1)
        
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        
        #proj_query中的第i行表示第i个像素位置上所有通道的值。size = B X N × C1
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) 
        
        #proj_key中的第j行表示第j个像素位置上所有通道的值，size = B X C1 x N
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) 
        
        #Energy中的第(i,j)是将proj_query中的第i行与proj_key中的第j行点乘得到
        #energy中第(i,j)位置的元素是指输入特征图第j个元素对第i个元素的影响，
        #从而实现全局上下文任意两个元素的依赖关系
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        
        #对行的归一化，对于(i,j)位置即可理解为第j位置对i位置的权重，所有的j对i位置的权重之和为1
        attention = self.softmax(energy) # B X N X N
        
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N
        out = torch.bmm(proj_value,attention.permute(0,2,1)) #B X C X N
        out = out.view(m_batchsize,C,width,height) #B X C X W X H
        
        #跨连，Gamma是需要学习的参数
        out = self.gamma*out + x #B X C X W X H
        
        return out,attention


def main():
    x = torch.randn(size = (4,16,20,20))  
    self_atten_spatial = Self_Attn_Spatial(16,4)
    y = self_atten_spatial(x)
    print('y.size:',y[0].size())   
    print y
    '''
    y.size: torch.Size([4, 16, 20, 20])
    '''
    

if __name__ == "__main__":
    main()

