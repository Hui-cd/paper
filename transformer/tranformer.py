import torch.nn as nn
import torch
import math
import copy
import torch.nn.functional as F

def attention(key,query,value,maske=None):
    key_t = key.transpose(-2,-1)
    d_k = query.size(-1)
    q_k = torch.matmul(key_t,query) 
    score = q_k/math.sqrt(d_k)
    x = F.softmax(score,dim=-1)
    return q_k,x


def clone(Model,N):
    """
    copy model
    """
    return nn.ModuleList([copy.deepcopy(Model) for _ in range(N)])

class MutilHeadAttention(nn.Module):
    def __init__(self,h,d_model,dropout) -> None:
        super(MutilHeadAttention,self).__init__()
        self.h = h
        self.d_model = d_model
        self.d_k = d_model // h
        self.atten = None
        self.linears = clone(nn.Linear(d_model,d_model),4)
        self.dropout = nn.Dropout(p=dropout)

        
    def forward(self,key,query,value):
        nbatches = query.size(0)
        query,key,value = [l(x).view(nbatches,-1,self.h,self.d_k)
            for l,x in zip(self.linears,(query,key,value))
        ]
        x, self.atten = attention(key=key,query=query,value=value)
        x = x.transpose(1,2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
    
    
class LayerNorm(nn.Module):
    def __init__(self) -> None:
        super(LayerNorm,self).__init__()
        
        
    def forward():
        pass
        
if __name__ == "__main__":
    key = torch.ones(2,2,3)
    print(key.size())