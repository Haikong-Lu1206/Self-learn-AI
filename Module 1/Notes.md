---

# Module 1：LLM 核心底层机制学习笔记

---

# 1️⃣ 数字化基石：Tokenization（BPE 算法）

## 1.1 核心知识点

### 🔹 Subword Tokenization

* 介于 **字符级** 与 **单词级** 之间
* 在 **词表大小** 和 **序列长度** 之间取得平衡
* 是现代 LLM 的主流做法

---

### 🔹 BPE（Byte Pair Encoding）

一种基于频率的压缩算法：

1. 从字符级开始
2. 统计最频繁的相邻字符对
3. 合并它们
4. 重复上述过程直到达到目标词表大小

👉 本质：**频率驱动的合并策略**

---

### 🔹 Byte-level Handling

* 文本被视为原始字节（0–255）
* 可以处理任意字符
* **彻底消除 OOV（Out-of-Vocabulary）问题**

---

## 1.2 互动问答回顾

### ❓ 为什么 BPE 处理数字（如 `"92837465"`）时计算能力较差？

**答：**

BPE 是基于频率合并的，可能会将数字拆成：

```
"92", "837", "465"
```

这会破坏数字的：

* 位值结构（Place Value）
* 数学规律

模型看到的是“字符串块”，而不是十进制结构。

---

### ❓ 如果一个词在训练语料中从未见过，BPE 会报错吗？

**答：不会。**

BPE 具有 **Robustness（鲁棒性）**：

* 会退化为单字节表示
* 逐字符处理
* 永远不会真正 OOV

---

# 2️⃣ 空间表征：Embedding & RoPE

---

## 2.1 核心知识点

### 🔹 Token Embedding

* 本质是一个查表（Lookup Table）
* 将离散 Token ID → 稠密向量
* 维度通常为 `d_model`

---

### 🔹 Positional Encoding（位置编码）

Transformer 是：

> Permutation Invariant（排列不变）

因此必须人为注入位置信息。

---

### 🔹 RoPE（Rotary Positional Embedding）

现代 LLM 标配。

核心思想：

* 在复数域中进行旋转
* 把绝对位置转换为向量之间的相对旋转关系

数学直觉：

```
位置 m 和 n 的关系 → 由旋转角度差 (m - n) 决定
```

---

## 2.2 互动问答回顾

### ❓ 为什么 RoPE 在超长文本时优于传统加法式 PE？

**答：**

RoPE 关注的是：

```
相对距离 (m - n)
```

这种几何关系在超出训练长度时依然成立。

而传统绝对位置编码：

```
PE(pos)
```

在未见过的位置会失效。

👉 RoPE 具有更好的 **外推性（Extrapolation）**

---

# 3️⃣ 现代架构：Transformer Block

---

## 3.1 核心知识点

### 🔹 RMSNorm

Root Mean Square Layer Normalization

相比 LayerNorm：

* 不减均值
* 只做缩放

优势：

* 更快
* 更稳定
* 更适合大模型

---

### 🔹 Pre-Norm 架构

结构：

```
x → Norm → Attention → Residual
```

而不是：

```
x → Attention → Norm
```

---

### 🔹 SwiGLU

结合：

* 门控机制（Gating）
* SiLU 激活函数

优于传统 ReLU：

* 更强非线性
* 更好表达能力

---

## 3.2 互动问答回顾

### ❓ 为什么现代模型偏好 Pre-Norm？

**答：**

Pre-Norm 允许：

```
梯度通过 Residual 直接流向底层
```

避免：

* 梯度消失
* 深层训练困难

---

# 4️⃣ 预测核心：Causal Masking

---

## 4.1 核心知识点

### 🔹 Autoregressive（自回归）

模型预测：

```
P(x_t | x_<t)
```

基于历史预测下一个 Token。

---

### 🔹 Causal Mask

使用：

```
下三角矩阵
```

在 Attention Score 中：

* 未来位置设为 `-∞`
* Softmax 后权重为 `0`

---

## 4.2 互动问答回顾

### ❓ 在 10×10 的 Attention 矩阵中，有多少元素被 Mask？

总元素：

```
100
```

保留（含对角线）：

```
55
```

被 Mask：

```
45
```

这体现了：

```
Attention 复杂度 = O(L²)
```

---

# 5️⃣ 推理优化：KV Cache

---

## 5.1 核心知识点

### 🔹 KV Cache

在生成阶段缓存：

```
前 N-1 个 Token 的 K 和 V
```

避免重复计算。

---

### 🔹 显存代价

KV Cache 占用：

```
O(L × Batch × Layers × Dim)
```

随序列长度线性增长。

---

## 5.2 互动问答回顾

### ❓ 存一个词（Dim=10, 32层, FP16）需要多少空间？

计算：

```
10 × 2 (K,V)
× 2 bytes (FP16)
× 32 layers
= 1280 Bytes
```

---

### ❓ Q 需要缓存吗？

**不需要。**

原因：

* Q 只代表当前时刻的查询
* 计算完即丢弃

---

# 6️⃣ 输出控制：Sampling Strategies

---

## 6.1 核心知识点

### 🔹 Temperature

控制 logits 平滑度：

```
logits / T
```

* T → 0 ：接近 Greedy
* T = 1 ：原始分布
* T > 1 ：更随机

---

### 🔹 Top-p（Nucleus Sampling）

步骤：

1. 按概率排序
2. 累计概率
3. 选择累计 ≥ p 的最小候选集
4. 在候选集中采样

特点：

* 动态候选集
* 比 Top-k 更灵活

---

## 6.2 互动问答回顾

### ❓ 如果 T → 0 会发生什么？

答案：

```
Greedy Search
```

永远选择概率最高的 Token。

---
