import torch 
import torch.nn as nn
from einops import rearrange, einsum
import math


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        ## nn.Linear(): 采用了相反的存储方法
        # self.weight = Parameter(
        #     torch.empty((out_features, in_features), **factory_kwargs)
        # )

        # 用于统一管理张量的设备和数据类型
        factory_kwargs = {'device': device, 'dtype': dtype}
        # 初始化 截断在三个标准差
        self.W = nn.Parameter(torch.empty((in_features,out_features),**factory_kwargs))
        std = math.sqrt(2.0 /(in_features + out_features))
        a = -3.0*std
        b = 3.0*std
        nn.init.trunc_normal_(self.W,0,std,a,b)

    @property
    def weight(self):
        return self.W

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.W, "... in_features, in_features out_features -> ... out_features")
    
class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()

        factory_kwargs = {'device': device, 'dtype': dtype}
        self.W = nn.Parameter(torch.empty((num_embeddings,embedding_dim),**factory_kwargs))
        std = 1
        a = -3.0*std
        b = 3.0*std
        nn.init.trunc_normal_(self.W,0,std,a,b)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # batch_size sequence_length, num_embeddings,embedding_dim -> batch_size, sequence_length, d_model
        return self.W[token_ids]
    
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.eps = eps
        self.d_model = d_model
        # initialize 1 
        self.gamma = nn.Parameter(torch.ones(d_model,**factory_kwargs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if(x.dtype != torch.float32):
            origin_dtype = x.dtype
            x = x.to(torch.float32)
            r_std = torch.rsqrt(torch.mean(x*x,dim=-1,keepdim=True) + self.eps)
            x = x*r_std
            y = x*self.gamma
            return y.to(origin_dtype)
        else:
            r_std = torch.rsqrt(torch.mean(x*x,dim=-1,keepdim=True) + self.eps)
            x = x*r_std
            y = x*self.gamma
            return y

class SiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x: torch.Tensor):
        return x*torch.sigmoid(x)
    
class FFN(nn.Module):
    def __init__(self, d_model : int, d_ff = None, multiple_of = 64,activation_func = 'SiLU', device=None, dtype=None):
        super().__init__()
        # factory_kwargs = {'device': device, 'dtype': dtype}
        self.activation_func = activation_func
        if d_ff is None:
            # 计算隐层维度 考虑到是正整数 直接取整即可
            d_ff = d_model*8//3
            d_ff = multiple_of * ((d_ff + multiple_of-1)//multiple_of)

        self.w_1 = Linear(d_model,d_ff,device,dtype)
        self.w_2 = Linear(d_ff,d_model,device,dtype)
        # GLU
        self.w_3 = Linear(d_ff,d_model,device,dtype)
        self.silu = SiLU()

    def forward(self,x: torch.Tensor):
        if(self.activation_func == 'SiLU'):
            return self.w_2(self.silu(self.w_1(x))*self.w_3(x))
        else:
            raise RuntimeError("no imple")

        
class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None, dtype=None):
        super().__init__()
        
        self.d_k = d_k
        # 先求下逆频率
        inv_freq = 1.0 / (torch.pow(theta, torch.arange(0,d_k,2)/d_k))
        inv_freq.to(device=device)

        # 位置
        positions = torch.arange(max_seq_len, device=device)
        # 外积 生成每个position的旋转频率
        freqs = torch.outer(positions, inv_freq)
        # 生成旋转矩阵 (max_seq_len, d_k/2)
        self.register_buffer("cos", freqs.cos(),persistent=False)
        self.register_buffer("sin", freqs.sin(),persistent=False)
        


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        """ 不失一般性的 x的shape应该是（ batch_size, seq_len, d_k) -> (batch_size, seq_len, d_k/2 ,2)"""
        x = rearrange(x, "... (d two) -> ... d two", two=2)
        # 生成x,y
        x1 = x[...,0]
        x2 = x[...,1]
        # (batch_size, seq_len, d_k/2)
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        
        rot_x1 = x1*cos - x2*sin
        rot_x2 = x1*sin + x2*cos

        rot_x = torch.stack([rot_x1, rot_x2], dim=-1)
        rot_x = rearrange(rot_x, "... d two -> ... (d two)", two=2)

        return rot_x.to(dtype=in_dtype)
        
def softmax(x: torch.Tensor, dim = -1):
    max_num = torch.max(x, dim,keepdim=True).values
    x = x - max_num
    exp_x = torch.exp(x)
    return exp_x / torch.sum(exp_x, dim=dim,keepdim=True)
    # pass

# class ScaledDotProductAttention(nn.Module):
#     def __init__(self, Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor):
#         super().__init__()

def scaled_dot_product_attention(Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor,mask= None):
    """ 不失一般性的 
        Q的shape应该是（ batch_size,..., seq_len_q, d_q) 
        K的shape应该是（ batch_size,..., seq_len_k, d_k) 
        V :  batch_size,..., seq_len_k, d_v)"""
    d_k = Q.shape[-1]
    # seq_len = Q.shape[-2]
    K_T = rearrange(K, "... seq_len_k d_k -> ... d_k seq_len_k")
    scores = einsum(Q,K_T,"... seq_len_q d_k, ... d_k seq_len_k -> ... seq_len_q seq_len_k")

    if(mask is not None):
        # assert mask.shape[-2:] == (seq_len,seq_len) mask有其他维度
        # 记得取反
        scores = scores.masked_fill(~mask,float('-inf'))
    scaled_scores = scores/math.sqrt(d_k)
    attn = softmax(scaled_scores,dim=-1)
    result = einsum(attn, V, "... seq_len_q seq_len_k, ... seq_len_k d_v -> ... seq_len_q d_v")
    return result

    
class MultiHeadsAttention(nn.Module):
    def __init__(self, d_model:int, num_heads:int,max_seq_len: int = None,theta: float = None, device=None, dtype=None):
        super().__init__()
        # 必要的检测
        assert d_model % num_heads == 0

        self.factory_kwargs = {'device': device, 'dtype': dtype}

        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
        # 要拆头的维度 以降低计算量
        # 为了并行化 这里直接d_model -> d_model 实际上应该竖着切成很多个
        self.w_q = Linear(d_model, d_model,device,dtype)
        self.w_k = Linear(d_model, d_model,device,dtype)
        self.w_v = Linear(d_model, d_model,device,dtype)

        self.w_o = Linear(d_model, d_model,device,dtype)
        self.rope = None
        if(max_seq_len and theta):
            self.rope = RoPE(theta,self.d_k,max_seq_len,device,dtype)

    def forward(self,x:torch.Tensor,token_positions = None):
        # pass
        assert x.shape[-1] == self.d_model
        seq_len = x.shape[-2]
        # 生成QKV
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)

        # 拆成多头
        Q = rearrange(Q, "batch_size seq_len (num_heads d_k) -> batch_size num_heads seq_len d_k", num_heads = self.num_heads)
        K = rearrange(K, "batch_size seq_len (num_heads d_k) -> batch_size num_heads seq_len d_k", num_heads = self.num_heads)
        if(self.rope is not None and token_positions is not None ):
            Q = self.rope(Q,token_positions)
            K = self.rope(K,token_positions)
        elif(self.rope and token_positions == None):
            token_positions = torch.arange(seq_len, device=x.device)
            Q = self.rope(Q,token_positions)
            K = self.rope(K,token_positions)
        V = rearrange(V, "batch_size seq_len (num_heads d_k) -> batch_size num_heads seq_len d_k", num_heads = self.num_heads)

        # 计算attn
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
        out = scaled_dot_product_attention(Q,K,V,mask)
        out = rearrange(out, "batch_size num_heads seq_len d_k -> batch_size seq_len (num_heads d_k)")
        return self.w_o(out)
