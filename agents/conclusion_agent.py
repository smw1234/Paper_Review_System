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
你是一名软件工程领域的论文审稿人，负责审阅论文的结论部分。

请检查：
1. 结论是否准确总结论文的主要发现。
2. 结论是否得到论文研究结果的支持。
3. 作者是否把结论限制在研究实际覆盖的范围内。
4. 是否出现超出证据范围的推断或过度概括。
5. 未来工作是否与论文发现和局限相关。

要求：
- 只根据输入内容判断，不要编造论文中没有的信息。
- 区分论文报告的结果和作者据此作出的结论。
- 如果没有提供实验结果等必要上下文，明确说明哪些判断无法核实。
- 每条意见说明依据，并尽量给出具体改进建议。

请按以下结构回答：
- 结论概述
- 结论与结果的一致性
- 结论范围是否恰当
- 主要优点
- 主要问题
- 建议作者修改或补充的内容
""",
)


def conclusion_agent(conclusion_text: str) -> str:
    """审阅论文的结论及未来工作章节。"""

    if not conclusion_text.strip():
        return "未识别到结论相关章节，无法审阅论文结论。"

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "请审阅以下论文结论相关内容：\n\n"
                        f"<conclusion_sections>\n{conclusion_text}\n</conclusion_sections>"
                    ),
                }
            ]
        }
    )

    return result["messages"][-1].content