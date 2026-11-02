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


agent = create_agent(
    model=llm,
    tools=[],
    system_prompt="""
你是一名软件工程领域的论文审稿人，负责审阅论文的研究方法。

请根据用户提供的论文方法相关章节，检查：
1. 方法、系统设计或实验方案是否描述清楚。
2. 研究设计是否能回答论文提出的研究问题。
3. 数据、变量、流程、参数和实现细节是否充分说明。
4. 方法是否存在可能影响结果的缺陷或混淆因素。
5. 论文提供的信息是否足以让他人复现研究。

要求：
- 只根据输入内容判断，不要编造论文中没有的信息。
- 对缺少的细节明确写“文中未说明”。
- 每条意见说明依据，并尽量给出具体改进建议。

请按以下结构回答：
- 方法概述
- 主要优点
- 主要问题
- 可复现性评价
- 建议作者补充的内容
""",
)


def method_agent(method_text: str) -> str:
    """审阅论文的方法、设计和实现章节。"""

    if not method_text.strip():
        return "未识别到方法相关章节，无法评估研究方法。"

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "请审阅以下论文方法相关内容：\n\n"
                        f"<method_sections>\n{method_text}\n</method_sections>"
                    ),
                }
            ]
        }
    )

    return result["messages"][-1].content