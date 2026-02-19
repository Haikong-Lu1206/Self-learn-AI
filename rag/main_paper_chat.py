import os
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
from retriever import SimpleRecursiveSplitter, SimpleRetriever
from reranker import SimpleReranker
from generate_prompt import generate_final_prompt
from dotenv import load_dotenv
from main_chat import call_deepseek

def main():
    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY").strip()
    print(api_key)
    # 1. 模拟或读取论文内容
    paper_text = """2030年交通蓝图：磁悬浮物流与低空飞行网。
在2030年的城市规划中，核心架构是名为“阿耳忒弥斯”（Artemis）的地下磁悬浮货运管道系统。
该系统由特斯拉动力实验室研发，运行压力维持在0.1个标准大气压。
管道内部的物流仓储机器人称为“标枪”（Javelin），它们能在时速350公里的情况下精准分拣包裹。
与此同时，低空领域开放给了民用垂直起降飞行器（eVTOL）。
这种飞行器采用的是固态锂硫电池，其能量密度达到了 500Wh/kg。
然而，该方案面临一个核心争议：由于锂硫电池在零下20摄氏度时放电效率会下降40%，导致北方城市的冬季续航能力受到质疑。
相比之下，南方城市（如深圳、海口）已经全面部署了该系统。
项目负责人李博文指出，解决低温问题的关键在于引入“石墨烯热管理夹层”。"""
    
    # 2. 准备组件
    splitter = SimpleRecursiveSplitter(chunk_size=200, chunk_overlap=30)
    chunks = splitter.split_text(paper_text)
    
    retriever = SimpleRetriever()
    retriever.build_index(chunks)
    
    reranker = SimpleReranker()
    
    while True:
        query = input("\n问：")
        if query == 'q': break
        
        # 3. 检索 + 重排
        results = retriever.query(query, k=5)
        re_chunks = [r["content"] for r in results]
        ranked_results = reranker.rerank(query, re_chunks)
        
        # 取最相关的 top 1 作为上下文
        context = ranked_results[0][0]
        
        # 4. 构造 Prompt
        final_prompt = f"参考资料：\n{context}\n\n问题：{query}"
        
        # 5. 调用 DeepSeek 获取答案
        print("\n[DeepSeek 思考中...]")
        answer = call_deepseek(final_prompt, api_key)
        
        print(f"\n答：{answer}")

if __name__ == "__main__":
    main()


