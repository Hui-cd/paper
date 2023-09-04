import torch.nn as nn
import torch

def attention(query,key,value,maske = None):
    d_k = key.size()
    score = torch.matmul(query,key.transpose)
    
    


if __name__ == "__main__":
    query = torch.ones(2, 4)
    key = torch.ones(2, 4)
    value = torch.ones(2, 4)
    
    result = key.transpose(-2,-1)
     
    # result = attention(query=query,key=key,value=value,maske=None)
    print(result)
