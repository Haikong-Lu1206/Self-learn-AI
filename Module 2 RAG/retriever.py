import faiss
import numpy as np
import re
from sentence_transformers import SentenceTransformer

class SimpleRecursiveSplitter:
    def __init__(self, chunk_size=100, chunk_overlap=20):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_text(self,text):
        sentences = re.split(r'(?<=[。！？\n])', text)
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk)+len(sentence)<=self.chunk_size:
                current_chunk+=sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) >= self.chunk_overlap else ""
                current_chunk = overlap_text + sentence
        if current_chunk:
            chunks.append(current_chunk)

        return chunks
class SimpleRetriever:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        # 1. 初始化 Embedding 模型
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []

    def build_index(self, documents):
        """
        将文档列表转化为向量并存入 FAISS 索引
        """
        self.documents = documents
        # 编码文档：将文本转为向量 (Batch 处理)
        embeddings = self.model.encode(documents, show_progress_bar=True)
        
        # 向量维度 (all-MiniLM-L6-v2 是 384)
        dimension = embeddings.shape[1]
        
        # 2. 构建 FAISS 索引 (IndexFlatL2 是暴力搜索，适合万级以下数据)
        self.index = faiss.IndexFlatL2(dimension)
        # FAISS 需要 float32 的 numpy 数组
        self.index.add(np.array(embeddings).astype('float32'))
        print(f"Index built with {len(documents)} documents.")

    def query(self, query_text, k=2):
        """
        查询最相关的 k 个文档
        """
        # 将查询词向量化
        query_vec = self.model.encode([query_text])
        
        # 3. 在索引中搜索
        # distances: 相似度距离 (L2 距离，越小越近)
        # indices: 文档在 self.documents 中的索引
        distances, indices = self.index.search(np.array(query_vec).astype('float32'), k)
        
        # 返回结果文档及其对应的距离
        results = [
            {"content": self.documents[idx], "score": dist} 
            for dist, idx in zip(distances[0], indices[0])
        ]
        return results

if __name__ == "__main__":
    text = "中子星是恒星演化到末期，经由引力坍缩发生超新星爆炸之后，可能成为的少数终点之一。中子星的密度极高，其半径仅有10-20公里，但质量却是太阳的1.35到2.1倍。"
    splitter = SimpleRecursiveSplitter(chunk_size=50, chunk_overlap=15)
    chunks = splitter.split_text(text)
    
    '''for i, c in enumerate(chunks):
        print(f"Chunk {i+1}: {c} (长度: {len(c)})")
    # --- 模拟知识库 ---
    knowledge_base = [
        "Gemini 是由 Google 开发的一款人工智能模型。",
        "PyTorch 是一个开源的机器学习库，广泛用于计算机视觉和自然语言处理。",
        "向量数据库（Vector DB）通过存储向量特征来实现语义搜索。",
        "红烧肉是一道经典的中国菜，主要原材料是五花肉。"
    ]'''
    knowledge_base = chunks

    retriever = SimpleRetriever()
    retriever.build_index(knowledge_base)

    # --- 测试查询 ---
    user_q = "中子星的半径以及质量？"
    hits = retriever.query(user_q, k=1)
    
    print(f"\n用户提问: {user_q}")
    for hit in hits:
        print(f"匹配结果: {hit['content']} (距离分数: {hit['score']:.4f})")