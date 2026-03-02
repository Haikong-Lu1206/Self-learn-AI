def generate_final_prompt(query, context):
    """
    构造注入了上下文的 Prompt
    """
    # 这里的模板是 RAG 的核心技巧：
    # 1. 明确角色 2. 约束来源 3. 处理未知情况
    prompt_template = f"""你是一个严谨的助手。请仅根据提供的【参考资料】来回答问题。
如果你在参考资料中找不到答案，请诚实地回答“资料中未提及此信息”，不要尝试编造。

【参考资料】：
{context}

【用户问题】：
{query}

【回答】："""
    return prompt_template