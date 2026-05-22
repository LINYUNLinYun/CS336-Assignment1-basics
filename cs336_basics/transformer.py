import torch 
import torch.nn as nn
from einops import rearrange, einsum
import math
from cs336_basics.module import RMSNorm, MultiHeadsAttention, FFN, Embedding,Linear,softmax

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len = None, theta = None, ):
        super().__init__()
        self.rms_norm1 = RMSNorm(d_model,)
        self.mha = MultiHeadsAttention(d_model, num_heads,max_seq_len,theta)
        self.ffn = FFN(d_model,d_ff)
        self.rms_norm2 = RMSNorm(d_model,)

    def forward(self,x : torch.Tensor):
        y = self.mha(self.rms_norm1(x)) + x
        out = self.ffn(self.rms_norm2(y)) + y
        return out
    
class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, num_layers: int,
                 d_model, num_heads, d_ff,  theta = None,):
        super().__init__()

        self.embd = Embedding(vocab_size, d_model,)
        self.layers = nn.ModuleList(
            TransformerBlock(d_model,num_heads,d_ff,context_length,theta,) 
            for _ in range(num_layers))
        self.rms_norm = RMSNorm(d_model)
        self.output_embd = Linear(d_model,vocab_size)
        
        

    
    def forward(self,x):
        x = self.embd(x)
        for layer in self.layers:
            x = layer(x)
        norm_x = self.rms_norm(x)
        return self.output_embd(norm_x)
        