import torch

def apply_rotary_pos_emb(x, m, theta=10000):
    """
    极简版二维旋转
    x: 输入向量 (Batch, Seq, Dim)
    m: 位置索引
    """
    B, L, D = x.shape
    # 假设我们只旋转前两个维度
    x_pairs = x[:, :, :2] 
    
    # 计算旋转角度
    angle = m / (theta ** (0 / D)) # 简化版 theta_i
    
    cos_m = torch.cos(torch.tensor(angle))
    sin_m = torch.sin(torch.tensor(angle))
    
    # 旋转矩阵操作
    x1 = x_pairs[:, :, 0]
    x2 = x_pairs[:, :, 1]
    
    # [x1*cos - x2*sin, x1*sin + x2*cos]
    rotated_x1 = x1 * cos_m - x2 * sin_m
    rotated_x2 = x1 * sin_m + x2 * cos_m
    
    return torch.stack([rotated_x1, rotated_x2], dim=-1)

# 测试
x = torch.ones(2, 3, 2) # 一个全 1 向量
print("原始向量:", x)
x_rot = apply_rotary_pos_emb(x, m=1) # 在位置 1 旋转
print("旋转后向量:", x_rot)