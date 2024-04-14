import torch
import torch.nn as nn


# class InfiniAttention(nn.Module):
#     def __init__(self,dim,inner_dim):
#         q_

# 12 layer  8 attention heads of dimension 128  
# FFn hidden layer 4096
# Infini-attention segment length N to 2048 
# for all attention layers and the input sequence length 
# to 32768 for trainin
def memory_retrieval(dim,inner_dim):
    query = nn.Linear(dim,inner_dim)
    key = nn.Linear(dim,inner_dim)
    value = nn.Linear(dim,inner_dim)
    nn.F.ELu