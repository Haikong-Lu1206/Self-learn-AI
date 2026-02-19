import torch
import torch.nn.functional as F
import torch.nn as nn
import math

def get_causal_mask(seq_len):
    # torch.tril 保留矩阵下三角部分(上三角形全是0)
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask

class MiniCausalAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # 1. 定义三个线性层，分别产生 Q, K, V
        # 假设我们现在不搞多头，只搞单头（Single Head）
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x 的形状是 (Batch_Size, Seq_Len, d_model)
        B, L, D = x.size()

        # 第一步：投影（Projection）
        q = self.q_proj(x) # (B, L, D)
        k = self.k_proj(x) # (B, L, D)
        v = self.v_proj(x) # (B, L, D)

        # 第二步：计算注意力分数（Attention Scores）
        # 公式：(Q @ K^T) / sqrt(dk)
        # transpose(-2, -1) 是把 K 的最后两维转置，方便做矩阵乘法
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(D) # 形状: (B, L, L)

        #device=x.device 确保生成的 mask 张量和输入 x 在同一个 CPU/GPU 上，
        #避免设备不匹配错误。
        mask = torch.tril(torch.ones(L, L, device=x.device))
       
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        # 第四步：归一化并与 V 相乘
        attn_weights = torch.softmax(attn_scores, dim=-1) # (B, L, L)
        output = attn_weights @ v # (B, L, D)

        return output
    
if __name__ == "__main__":
    # 模拟：1个样本, 序列长度5, 向量维度32
    x = torch.randn(1, 5, 32)
    model = MiniCausalAttention(32)
    out = model(x)
    print("输出形状:", out.shape) # 应该是 (1, 5, 32)
    