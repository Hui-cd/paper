import torch
import torch.nn as nn


class MultiHeadInfiniAttention(nn.Module):
    def __init__(self,n_head,dim_input,dim_k,dim_q,dim_v,segment_length):
        super(MultiHeadInfiniAttention).__init__()
        self.dim_input = dim_input
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dim_q = dim_q
        self.segment_length = segment_length
        self.beta = nn.Parameter()
        w_q = nn.Linear(dim_input,dim_q*n_head)
        w_k = nn.Linear(dim_input,dim_k*n_head)
        w_v = nn.Linear(dim_input,dim_v*n_head)
        
    def memory_retrieval(self,memory,z,q):
        segma_q = nn.ELU(q)
        a_mem = torch.matmul(segma_q,memory)/torch.matmul(segma_q,z)
    
    def memory_update(self,k,v):
        memory = torch.zeros(1,self.dim_k,self.dim_v)
        z = torch.zeros(1,self.dim_k)
        segma_k = nn.ELU(k)
        delta_rule = v - torch.matmul(segma_k,memory)/torch.matmul(segma_k,z)
        memory = memory+ torch.matmul(nn.ELU(k).T,delta_rule)
        z = z + nn.ELU(k).sum()
        
        return memory,z
        
    def long_term_context_injection(self,a_mem,a_dot):
        a = nn.Sigmoid(self.beta)*a_mem + (1 - nn.Sigmoid(self.beta))*a_dot
        return a
    
        
        
        
                 

# 12 layer  8 attention heads of dimension 128  
# FFn hidden layer 4096
# Infini-attention segment length N to 2048 
# for all attention layers and the input sequence length 
# to 32768 for trainin
