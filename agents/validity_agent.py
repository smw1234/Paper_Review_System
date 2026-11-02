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
你是一名软件工程领域的论文审稿人，负责审阅研究的有效性威胁和局限性。

请检查：
1. 作者是否讨论了研究结果可能不成立或不适用的条件。
2. 是否考虑内部有效性、外部有效性、构念有效性和结论有效性。
3. 数据集、样本、测量方式、实验设置或分析方法是否可能影响结论。
4. 作者对研究结果的适用范围和限制是否说明清楚。
5. 论文是否提出了减轻这些威胁的措施。

要求：
- 只根据输入内容判断，不要编造论文中没有的信息。
- 区分作者明确承认的局限和你发现的潜在威胁。
- 文中没有讨论某项威胁时，说明“提供的内容中未讨论”，不要直接断定全文没有讨论。
- 每条意见说明依据，并给出具体改进建议。

请按以下结构回答：
- 作者明确说明的局限
- 潜在的有效性威胁
- 已采取的缓解措施
- 建议作者补充的内容
""",
)


def validity_agent(validity_text: str) -> str:
    """审阅论文的有效性威胁和研究局限。"""

    if not validity_text.strip():
        return "未收到有效性或局限性相关内容，无法进行评估。"

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "请审阅以下论文中与有效性威胁和局限性有关的内容：\n\n"
                        f"<validity_sections>\n{validity_text}\n</validity_sections>"
                    ),
                }
            ]
        }
    )

    return result["messages"][-1].content