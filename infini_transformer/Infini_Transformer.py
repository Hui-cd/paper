import torch
import torch.nn as nn
import torch.nn.functional as F  # Import functional module

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadInfiniAttention(nn.Module):
    def __init__(self, n_head, dim_input, dim_k, dim_v, segment_length):
        super(MultiHeadInfiniAttention, self).__init__()
        self.n_head = n_head
        self.dim_input = dim_input
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.segment_length = segment_length
        self.beta = nn.Parameter(torch.zeros(1))

        self.w_q = nn.Linear(dim_input, n_head * dim_k)
        self.w_k = nn.Linear(dim_input, n_head * dim_k)
        self.w_v = nn.Linear(dim_input, n_head * dim_v)

    def forward(self, x):
        batch_size, sequence_len, _ = x.shape
        n_seq, rem = divmod(sequence_len, self.segment_length)
        if rem != 0:
            raise ValueError("Sequence length must be divisible by the segment length.")

        memory = torch.zeros(batch_size, self.n_head, self.dim_k, self.dim_v, device=x.device)
        z = torch.zeros(batch_size, self.n_head, self.dim_k, 1, device=x.device)

        outputs = []
        for i in range(n_seq):
            start = i * self.segment_length
            end = start + self.segment_length
            segment = x[:, start:end, :]

            k = self.w_k(segment).view(batch_size, -1, self.n_head, self.dim_k).transpose(1, 2)
            q = self.w_q(segment).view(batch_size, -1, self.n_head, self.dim_k).transpose(1, 2)
            v = self.w_v(segment).view(batch_size, -1, self.n_head, self.dim_v).transpose(1, 2)

            memory, z = self.memory_update(k, v, memory, z)
            
            # Retrieve memory-based attention and apply scaled dot-product attention
            a_mem = self.memory_retrieval(memory, z, q)
            a_dot = F.softmax(q @ k.transpose(-2, -1) / torch.sqrt(torch.tensor(self.dim_k)), dim=-1) @ v
            
            # Integrate both types of attention using long-term context injection
            combined_attention = self.long_term_context_injection(a_mem.mean(dim=1), a_dot.mean(dim=1))
            outputs.append(combined_attention)

        # Concatenate all segment outputs
        return torch.cat(outputs, dim=1)

    def memory_retrieval(self, memory, z, q):
        sigma_q = F.elu(q) + 1.0
        a_mem = torch.matmul(sigma_q, memory) / (torch.matmul(sigma_q, z.unsqueeze(-1)) + 1e-5)
        return a_mem
    
    def memory_update(self, k, v, memory, z):
        sigma_k = F.elu(k) + 1.0
        sigma_k_transposed = sigma_k.transpose(-2, -1)  # Align dimensions for multiplication
        memory_update = torch.matmul(sigma_k_transposed, v)
        memory += memory_update
        summed_sigma_k = sigma_k.sum(dim=-2)
        print("Shape of z:", z.shape)
        print("Shape of summed_sigma_k:", summed_sigma_k.shape)
        print("self.segment_length:", self.segment_length)
        z += summed_sigma_k.unsqueeze(-1) * self.segment_length  # Adjusting dimension if necessary
        return memory, z

    def long_term_context_injection(self, a_mem, a_dot):
        beta_sigmoid = torch.sigmoid(self.beta)
        combined_attention = beta_sigmoid * a_mem + (1 - beta_sigmoid) * a_dot
        return combined_attention


if __name__ == '__main__':
    n_head = 8
    dim_input = 512
    dim_key = 64
    dim_value = 64
    segment_len = 32
    model = MultiHeadInfiniAttention(n_head=n_head, dim_input=dim_input, dim_k=dim_key, segment_length=segment_len, dim_v=dim_value)
    test = torch.randn(4, 128, dim_input)
    x = model(test)

        
                 

# 12 layer  8 attention heads of dimension 128  
# FFn hidden layer 4096
# Infini-attention segment length N to 2048 
# for all attention layers and the input sequence length 
# to 32768 for trainin
