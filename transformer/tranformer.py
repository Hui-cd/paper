import torch.nn as nn
import torch
import math
import copy
import torch.nn.functional as F

def attention(key,query,value,maske=None):
    key_t = key.transpose(-2,-1)
    d_k = query.size(-1)
    score = torch.matmul(key_t,query)/math.sqrt(d_k)
    return score


def clones(module, N):
    """
    生成N个相同的层
    :param module:(nn.Module)输入模型
    :param N:(int)重复次数
    :return: 复制生成的模型列表
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class MutilHeadAttention(nn.Module):
    def __init__(self,h,d_model) -> None:
        super(MutilHeadAttention,self).__init__()
        self.h = h
        
    def forward():
        
if __name__ == "__main__":
    key = torch.ones(2,2,3)
    print(key.size())