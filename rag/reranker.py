from sentence_transformers import CrossEncoder

class SimpleReranker:
    def __init__(self, model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
        # 这是一个专门训练用来判断相关性的模型
        self.model = CrossEncoder(model_name)

    def rerank(self, query, retrieved_chunks):
        # 1. 构造 (Query, Chunk) 配对
        pairs = [[query, chunk] for chunk in retrieved_chunks]
        
        # 2. 计算相关性得分 (0 到 1 之间)
        scores = self.model.predict(pairs)
        
        # 3. 按得分排序
        results = sorted(zip(retrieved_chunks, scores), key=lambda x: x[1], reverse=True)
        return results

# --- 测试重排逻辑 ---
if __name__ == "__main__":
    query = "秦始皇在哪里称帝？"
    # 模拟 retriever 找回来的两个看起来很像的块
    chunks = [
        "秦始皇出生于赵国首都邯郸，他是庄襄王之子。", # 含有秦始皇，但地点不对
        "秦始皇在咸阳举行了盛大的称帝仪式，建立了统一的中央集权国家。" # 这才是正确答案
    ]
    
    reranker = SimpleReranker()
    ranked_results = reranker.rerank(query, chunks)
    
    for chunk, score in ranked_results:
        print(f"得分: {score:.4f} | 内容: {chunk}")