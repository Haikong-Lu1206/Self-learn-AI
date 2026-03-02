import torch
import torch.nn.functional as F

def temperature_sample(logits, temperature=1.0):
    if temperature == 0:
        return torch.argmax(logits, dim=-1)
    
    # 应用温度
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    # 多项式分布采样
    return torch.multinomial(probs, num_samples=1)