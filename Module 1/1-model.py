import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # 计算均方根时保持维度，利用 rsqrt 提升效率
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * rms) * self.weight

class SwiGLU(nn.Module):
    """
    现代大模型标配的 FFN 结构
    SwiGLU(x) = (Swish(xW) * xV) * W_down
    """
    def __init__(self, d_model: int, intermediate_size: int = None):
        super().__init__()
        intermediate_size = intermediate_size or int(d_model * 8 / 3) # 常见的隐藏层缩放
        self.w1 = nn.Linear(d_model, intermediate_size, bias=False) # Gate 路径
        self.w2 = nn.Linear(d_model, intermediate_size, bias=False) # Value 路径
        self.w3 = nn.Linear(intermediate_size, d_model, bias=False) # Down-projection

    def forward(self, x):
        # 对应公式: Swish(xW1) * xW2
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class CausalAttention(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False) # 增加输出投影

    def forward(self, x, mask=None, past_kv=None):
        B, L, D = x.size()
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # --- KV Cache 逻辑 ---
        if past_kv is not None:
            k_old, v_old = past_kv
            k = torch.cat([k_old, k], dim=1) # 拼接历史 Key
            v = torch.cat([v_old, v], dim=1) # 拼接历史 Value
        
        present_kv = (k, v)
        
        # Scaled Dot-Product Attention
        # 注意：这里的 q 可能只有当前 token (L=1)，但 k, v 包含历史
        attn_weights = (q @ k.transpose(-2, -1)) / math.sqrt(D)
        
        if mask is not None:
            # mask 需要能处理广播机制
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
        
        probs = F.softmax(attn_weights, dim=-1)
        output = probs @ v
        
        return self.o_proj(output), present_kv

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalAttention(d_model)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model) 
        
    def forward(self, x, mask=None, past_kv=None):
        # 1. Pre-norm Attention + Residual
        # x 先过 Norm，再进 Attention，最后加回原始 x
        norm_x = self.norm1(x)
        attn_out, kv = self.attn(norm_x, mask, past_kv)
        x = x + attn_out
        
        # 2. Pre-norm FFN + Residual
        x = x + self.ffn(self.norm2(x))
        
        return x, kv