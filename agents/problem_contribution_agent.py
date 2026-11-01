import os
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model=os.environ.get("DEEPSEEK_MODEL", "deepseek-flash"),
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url=os.environ.get(
        "DEEPSEEK_BASE_URL",
        "https://api.deepseek.com",
    ),
)

system_prompt = """
你是一名软件工程领域的论文审稿人。
请根据输入的论文内容，审阅研究问题、研究动机和论文贡献。

请遵守以下要求：
1. 只根据输入内容判断，不要编造信息。
2. 分析研究问题是否清楚、重要，动机是否充分。
3. 分析作者声称的贡献是否清楚，以及论文是否说明了与相关工作的区别。
4. 如果当前内容不足以判断，请明确写出“无法根据提供的章节判断”。
5. 说明判断依据，不要只给结论。

请按以下结构回答：
- 问题与动机
- 贡献清晰度
- 与相关工作的区别
- 主要优点
- 主要疑问
- 尚无法判断的内容
"""

# 创建“问题与贡献审稿” agent
agent = create_agent(
    model=llm,
    tools=[],
    system_prompt= system_prompt,)
def problem_contribution_agent(paper_text:str)->str:
    """审阅论文的问题定义、研究动机和贡献。"""
    if not paper_text.strip():
        return  "未能识别相关章节，无法评估论文的问题与贡献。"
    response=agent.invoke(
        {"messages": [
            {
                "role": "user",
                "content": (
                    "请审阅以下论文内容：\n\n"
                    f"<paper_sections>\n{paper_text}\n</paper_sections>"
                ),
            }
        ]}
    )
    return response["messages"][-1].content











