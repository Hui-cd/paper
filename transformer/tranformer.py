import torch.nn as nn
import torch
import math

def attention(query,key,value,mask = None):
    d_k = key.size(-1)
    score = torch.matmul(query,key.transpose(-2,-1))/math.sqrt(d_k)
    if mask is not None:
       score = score.masked_fill(mask==0,1e9) 
    return score    
    
def mask_attention(score):
    pass
    

if __name__ == "__main__":
    query = torch.ones(2, 4,3)
    key = torch.ones(2, 4)
    value = torch.ones(2, 4)
    
    # result = key.transpose(-2,-1)
    tensor1 = torch.randn(4, 5,5)
    tensor2 = torch.randn(4, 5)
    
    # result = torch.matmul(tensor1, tensor2).size() 
    # result = attention(query=tensor1,key=tensor2,value=value,mask=None)
    print(query)
    print(query.transpose(0,1))
    # print(result)
