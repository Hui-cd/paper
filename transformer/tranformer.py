import torch.nn as nn
import torch
import math
import copy
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self):
        pass 

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model, heads):
        super(MultiHeadAttention,self).__init__()
        self.heads = heads
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model,d_model)
        self.w_k = nn.Linear(d_model,d_model)
        self.w_v = nn.Linear(d_model,d_model)
        self.w_concat = nn.Linear(d_model,d_model)
    
    def forward(self,q,k,v,mask=None):
        q,k,v = self.w_q(q) ,self.w_k(k),self.w_v(v)
        q, k, v = self.split(q), self.split(k), self.split(v)
        out,attention = self.attention(q,k,v,mask=mask)
        
        out = self.concat(out)
        out = self.w_concat(out)
        
        return out
    
    def split(self,tensor):
        batch_size,length,d_model = tensor.size()
        
        d_tensor = d_model// self.heads
        tensor = tensor.view(batch_size,length,self.n_head,d_tensor).transpose(1,2)
        
        return tensor
    
    def concat(self,tensor):
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor
        
        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        
        return tensor
    
class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProductAttention).__init__()
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self,q,k,v,mask=None):
        batch_size,head, sequence_length, d_model = k.size
        k_t = k.transpose(2,3)
        score = (q@k_t)/math.sqrt(d_model)
        
        if mask is not None:
            score = score.masked_fill(mask==0,-1000)
        score = self.softmax(score)
        v = score@v
        return v,score
    
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        # '-1' means last dimension. 

        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out

class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
   
class EncoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, src_mask):
        # 1. compute self attention
        _x = x
        x = self.attention(q=x, k=x, v=x, mask=src_mask)
        
        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)
        
        # 3. positionwise feed forward network
        _x = x
        x = self.ffn(x)
      
        # 4. add and norm
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x