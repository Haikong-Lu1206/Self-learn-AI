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

# 📁 代码库结构清单

```
core/
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


# Module 2：Full RAG Pipeline（检索增强生成）笔记

---

# 1️⃣ 核心概念与原理

---

## 1.1 向量数据库底层（Vector Database）

### 🔹 Embedding

* 将文本映射为高维空间中的向量坐标
* 语义相近 → 空间距离更近
* 语义不相关 → 空间距离更远

> 本质：把“语言”转成“几何问题”

---

### 🔹 相似度度量（Similarity Metrics）

#### 1️⃣ Cosine Similarity（余弦相似度）

* 衡量向量夹角
* 对长度不敏感
* RAG 最常用

公式：

[
\cos(\theta) = \frac{A \cdot B}{||A|| ||B||}
]

---

#### 2️⃣ Euclidean Distance（L2 距离）

* 衡量两点之间的直线距离
* 对向量长度敏感

---

### 🔹 近似最近邻算法（ANN）

用于在百万/亿级数据中快速搜索。

---

#### ✅ HNSW（Hierarchical Navigable Small World）

* 基于多层图结构
* 类似“高速公路 + 本地道路”导航
* 查询速度极快
* 工业界主流方案

---

#### ✅ IVFFlat

* 先对向量空间进行聚类
* 搜索时只扫描最近的几个簇
* 牺牲一点精度换取速度

---

## 1.2 分块策略（Chunking）

---

### 🔹 为什么需要分块？

#### 1️⃣ Context Window 限制

Embedding 模型有最大输入长度。

#### 2️⃣ 语义稀释

文档太长 → 向量表达会模糊主题。

---

### 🔹 常见分块方法

---

#### ✅ Recursive Character Splitting

优先级：

1. 段落
2. 句子
3. 字符

保留原有文本结构，避免粗暴截断。

---

#### ✅ Overlap（重叠）

* 在相邻块之间保留一部分重复内容
* 防止语义断裂
* 解决“主语丢失”问题

---

#### ✅ Semantic Chunking

* 根据语义转折点动态切分
* 通过 Embedding 差异判断分割点
* 更智能但计算成本更高

---

# 2️⃣ 检索架构：Two-Stage Retrieval

---

## 2.1 Bi-Encoder（粗排）

特点：

* 文档 Embedding 预先计算
* Query 单独编码
* 向量空间相似度搜索

适用场景：

* 百万级数据快速筛选
* 获取 Top-K 候选

常用工具：

* `faiss`
* `chromadb`

---

## 2.2 Cross-Encoder（精排 / Reranker）

特点：

* 同时输入 `(Query + Chunk)`
* 直接计算交互得分

优点：

* 精度极高
* 能识别：

  * 否定词
  * 主被动关系
  * 逻辑细节

缺点：

* 速度慢
* 只能处理小规模候选集

---

### ✅ 工业级最佳实践

```
大规模数据
    ↓
Bi-Encoder（Top 50）
    ↓
Cross-Encoder（Top 5）
    ↓
送入 LLM
```

平衡：

* 速度
* 精度
* 成本

---

# 3️⃣ 高级 RAG 技巧

---

## 3.1 HyDE（Hypothetical Document Embeddings）

---

### 🔹 流程

```
Query
  ↓
LLM 生成“假设答案”
  ↓
对假设答案做 Embedding
  ↓
检索真实文档
```

---

### 🔹 原理

虚构答案通常：

* 句式完整
* 包含术语
* 接近真实文档表达方式

Embedding 匹配的是：

> 表达分布（Distribution），而不是事实真假。

因此即使答案错误，也能提升召回率。

---

## 3.2 Prompt 注入（Augmentation）

---

### 🔹 基本原则

* 明确角色（System Prompt）
* 明确来源限制（只基于 Context）
* 明确未知处理方式（“如果不知道，请回答不知道”）

---

### 🔹 标准结构

```
[System Prompt]
[Context Fragments]
[User Query]
```

---

# 4️⃣ 问答回顾（Q&A）

---

### ❓ Q1：为什么不能直接用 LLM 内部的 Token Embedding 做 RAG 检索？

**答：**

* LLM 内部是 Token 级别向量
* RAG 检索是段落级语义
* 简单平均会：

  * 丢失语序
  * 丢失结构
  * 无法表达整体语义

因此需要专门的 Sentence / Document Embedding 模型。

---

### ❓ Q2：Chunk Size=500，Overlap=0 会有什么问题？

如果答案刚好位于两个块的边界：

* 单个块信息不完整
* 检索失败
* 或生成碎片化回答

Overlap 是防止语义断裂的关键。

---

### ❓ Q3：为什么不直接用 Cross-Encoder 搜索整个数据库？

因为：

```
Cross-Encoder 需要对每个 (Query, Doc) 计算交互
```

计算复杂度：

[
O(N)
]

在百万级数据下不可行。

所以：

* 先用 Bi-Encoder 粗筛
* 再用 Cross-Encoder 精排

---

### ❓ Q4：HyDE 生成的答案是错的，为什么还能生效？

因为 Embedding 匹配的是：

```
表达模式
```

而不是事实真假。

虚构答案：

* 句式完整
* 含专业术语
* 分布更接近真实文档

因此更容易在向量空间中靠近目标文档。

---

# 5️⃣ 动手实战代码（核心片段）

---

## 5.1 递归切分（Python）

```python
def recursive_split(text, chunk_size=200, overlap=30):
    """
    核心思想：
    - 保留上一块末尾 overlap 进入下一块开头
    - 优先识别句子边界
    - 防止语义被截断
    """
```

---

## 5.2 接入 DeepSeek 做 RAG

使用 OpenAI 兼容接口：

```python
from openai import OpenAI

client = OpenAI(
    api_key="YOUR_API_KEY",
    base_url="https://api.deepseek.com"
)

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {
            "role": "user",
            "content": f"Context: {context}\nQuery: {query}"
        }
    ]
)

print(response.choices[0].message.content)
```

---

# 🧠 RAG 全流程总结图

```
文档
  ↓
Chunking
  ↓
Embedding
  ↓
向量数据库（HNSW / IVFFlat）
  ↓
Query
  ↓
Bi-Encoder 检索
  ↓
Cross-Encoder 精排
  ↓
Prompt 注入
  ↓
LLM 生成答案
```

---

# 🎯 核心理解总结

RAG 本质是：

> 把语言问题转化为几何问题（向量检索）
> 再转回语言问题（生成答案）

它的核心不只是“检索”，而是：

* 分布匹配
* 表达增强
* 上下文约束
* 精度与效率平衡

---
