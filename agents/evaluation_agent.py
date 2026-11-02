import os

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI


llm = ChatOpenAI(
    model=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url=os.environ.get(
        "DEEPSEEK_BASE_URL",
        "https://api.deepseek.com",
    ),
)


agent = create_agent(
    model=llm,
    tools=[],
    system_prompt="""
你是一名软件工程领域的论文审稿人，负责审阅论文的实验设计、评估过程和结果。

请检查：
1. 实验或评估是否能回答论文提出的研究问题。
2. 数据集、样本、基线、指标和实验设置是否交代清楚。
3. 结果是否充分支持作者的结论，是否存在过度解读。
4. 对比是否公平，结果是否报告完整。
5. 是否提供了必要的统计分析、误差分析或稳健性分析。

要求：
- 只根据输入内容判断，不要编造论文中没有的信息。
- 对文中没有交代的细节明确写“文中未说明”。
- 每条意见说明依据，并给出具体改进建议。
- 区分论文明确报告的事实和你的审稿判断。

请按以下结构回答：
- 评估内容概述
- 主要优点
- 主要问题
- 结果是否支持论文结论
- 建议作者补充的内容
""",
)


def evaluation_agent(evaluation_text: str) -> str:
    """审阅论文的实验、评估和结果章节。"""

    if not evaluation_text.strip():
        return "未识别到实验或评估相关章节，无法评价实验结果。"

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "请审阅以下论文实验、评估和结果相关内容：\n\n"
                        f"<evaluation_sections>\n{evaluation_text}\n</evaluation_sections>"
                    ),
                }
            ]
        }
    )

    return result["messages"][-1].content