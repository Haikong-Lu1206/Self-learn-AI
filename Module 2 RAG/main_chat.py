from openai import OpenAI

def call_deepseek(prompt, api_key):
    # 初始化客户端，注意 base_url 必须指向 DeepSeek 的服务器
    client = OpenAI(
        api_key=api_key, 
        base_url="https://api.deepseek.com"
    )

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",  # 或者用 deepseek-reasoner (即 R1 模型)
            messages=[
                {"role": "system", "content": "你是一个基于提供资料回答问题的严谨助手。"},
                {"role": "user", "content": prompt},
            ],
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"API 调用出错: {e}"