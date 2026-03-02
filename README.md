# 🚀 LLM-From-Scratch & RAG Pipeline  (Implementation)
本项目包含两个核心模块：从零构建的大语言模型（LLM）底层架构，以及基于工业级链路的检索增强生成（RAG）系统。旨在通过底层算法实现，深刻理解 Transformer 推理全流程与现代知识库问答机制。
# Module 1：LLM 核心底层机制学习笔记 ✅

---
# 📁 Module 1 代码库结构清单

```
Module 1/
├── 1-tokenizer.py   # BPE 训练、编码、解码
├── 1-model.py       # RMSNorm、CausalAttention、TransformerBlock
├── 1-Sampling.py    # 基于温度的采样函数
├── 1-RoPE.py         # RoPE 实现
├── Notes.md          # 学习笔记：核心概念、原理、互动问答
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
# 📁 Module 2 RAG 代码库结构清单 ✅

```
Module 2 RAG/
├── embedding_test.py    # 语义空间验证：测试不同句子的余弦相似度
├── retriever.py         # 检索内核：基于 FAISS 的向量索引与递归文本切片
├── reranker.py          # 重排序模块：引入 Cross-Encoder 修正语义检索偏差
├── generate_prompt.py   # 提示词工程：构建上下文注入模板
├── main_chat.py         # 基础问答：LLM API 调用与流式交互逻辑
├── main_paper_chat.py   # 集成实战：针对长文本/论文的完整问答系统
├── Notes.md          # 学习笔记：核心概念、原理、互动问答
└── .env                 # 环境配置：敏感 API Keys 隔离 (已加入 .gitignore)

```

# 💡 技术亮点
两阶段检索 (Retrieve & Rerank)：

粗筛：使用 all-MiniLM-L6-v2 (Bi-Encoder) 在 FAISS 索引中进行高效向量搜索。

精排：引入 ms-marco-MiniLM (Cross-Encoder) 对候选块进行深度打分，极大提升了回答的精准度。

递归分块 (Recursive Chunking)：

采用 SimpleRecursiveSplitter 策略，配合 overlap 重叠机制，确保上下文在切分处语义不丢失。

# 项目未来路线图 (Roadmap)
本项目计划持续迭代，从底层算子进化为具备多模态能力的自主智能体。

## 🤖 Module 3: AI Agent Architecture (智能体架构) 🚧
**目标**：从“被动问答”进化为“主动执行”。
- **Reasoning**：实现 ReAct (Reason + Act) 循环与 CoT (Chain of Thought) 推理流程。
- **Tool Use**：基于 JSON Schema 的 Function Calling 协议解析。
- **Memory**：构建 Window-based 短期记忆与基于向量数据库的长期记忆系统。
- **Planning**：复杂任务拆解与状态机管理（基于 LangGraph 思想）。
- **Project**：🚀 "Autonomous Researcher" —— 能够自主搜索、总结并撰写报告的调研助手。

## 👁️ Module 4: Multimodal Foundations (多模态基础)
**目标**：为模型装上“眼睛”，实现图文对齐。
- **ViT (Vision Transformer)**：图像 Patch Embedding 与注意力机制实现。
- **CLIP**：学习 Contrastive Loss，实现图文跨模态对齐。
- **LLaVA Architecture**：研究线性投影层（Linear Projector）如何将图像特征输入 LLM。
- **Project**：🖼️ "Vision-RAG" —— 针对图像内容的知识库检索与视觉问答系统。

## ⚡ Module 5: Fine-Tuning & Deployment (微调与部署)
**目标**：模型专业化定制与工业级发布。
- **PEFT**：深入 LoRA (Low-Rank Adaptation) 数学原理：$W = W_0 + BA$。
- **Quantization**：研究 QLoRA 与 NF4 量化细节，实现消费级显卡微调。
- **Serving**：学习 vLLM 与 PagedAttention，优化高并发推理性能。
- **Project**：🏁 "Omni-Assistant" —— 整合 RAG、视觉与 Agent 能力的最终形态全能助手。