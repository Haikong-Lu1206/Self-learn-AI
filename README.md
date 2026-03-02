---

# Module 1：LLM 核心底层机制学习笔记

---
# 📁 Module 1 代码库结构清单

```
Module 1/
├── 1-tokenizer.py   # BPE 训练、编码、解码
├── 1-model.py       # RMSNorm、CausalAttention、TransformerBlock
├── 1-Sampling.py    # 基于温度的采样函数
├── 1-RoPE.py         # RoPE 实现
```

---

# 🧠 总结结构图

```
Tokenization
   ↓
Embedding + RoPE
   ↓
Transformer Block × N
   ↓
Causal Mask
   ↓
KV Cache（推理优化）
   ↓
Sampling（输出控制）
```

---


# Module 2：Full RAG Pipeline（检索增强生成）结构清单



