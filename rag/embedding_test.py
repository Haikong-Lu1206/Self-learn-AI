from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
import torch

# 1. 加载模型 (首次运行会自动下载权重，约 80MB)
# all-MiniLM-L6-v2 是一个极小且高效的模型，适合做语义搜索
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. 准备句子
sentences = [
    "我喜欢猫",           # 目标句
    "狗是人类的好朋友",    # 潜在相关句 (都是宠物)
    "今天要出差"         # 无关句
]

# 3. 转化为向量 (Embeddings)
# 输出是一个 [3, 384] 的矩阵，每一行代表一个句子的向量（维度为 384）
embeddings = model.encode(sentences, convert_to_tensor=True)

# 4. 计算余弦相似度 (Cosine Similarity)
# 我们拿第一句 (index 0) 分别和另外两句对比
# Cosine Similarity 公式: (A · B) / (||A|| * ||B||)
sim_cat_dog = F.cosine_similarity(embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0))
sim_cat_weather = F.cosine_similarity(embeddings[0].unsqueeze(0), embeddings[2].unsqueeze(0))
# unsqueeze(0) 是为了把向量从 [384] 变成 [1, 384]，以符合 cosine_similarity 的输入要求
print(f"句子 1: {sentences[0]}")
print("-" * 30)
print(f"与 '{sentences[1]}' 的相似度: {sim_cat_dog.item():.4f}")
print(f"与 '{sentences[2]}' 的相似度: {sim_cat_weather.item():.4f}")
# .item是将torch.Tensor对象转换为Python的数值类型（如float或int），以便更方便地进行打印或其他操作。
# 5. 验证直觉
if sim_cat_dog > sim_cat_weather:
    print("\n结论: 向量空间成功识别出 '猫' 与 '狗' 的语义关联度高于 '天气'!")