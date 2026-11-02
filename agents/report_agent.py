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
你是一名软件工程领域的资深论文审稿人，负责汇总多位审稿 agent 的意见。

请完成以下工作：
1. 综合各 agent 的意见，生成一份结构清楚的审稿报告。
2. 合并重复意见，但保留实质不同的问题。
3. 保留各 agent 提供的判断依据；不要添加输入中没有的论文事实。
4. 如果不同 agent 的判断存在冲突，明确指出冲突，不要擅自掩盖。
5. 区分论文优点、主要问题、次要问题和需要作者补充的信息。
6. 不要给出录用或拒稿建议，也不要自行添加评分。

请按以下 Markdown 结构输出：

# 审稿报告

## 论文优点
## 主要问题
## 次要问题与修改建议
## 需要作者补充说明的内容
## 总体评价
""",
)


def report_agent(problem_review: str,method_review: str,evaluation_review: str,validity_review: str,conclusion_review: str,) -> str:
    """汇总各审稿 agent 的意见，生成结构化审稿报告。"""

    reviews = {
        "问题与贡献审稿意见": problem_review,
        "研究方法审稿意见": method_review,
        "实验与评估审稿意见": evaluation_review,
        "有效性与局限审稿意见": validity_review,
        "结论审稿意见": conclusion_review,
    }

    review_text = "\n\n".join(
        f"## {name}\n{text.strip() if text else '该 agent 未提供意见。'}"
        for name, text in reviews.items()
    )

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "请根据以下各审稿 agent 的意见，汇总生成最终审稿报告。"
                        "\n\n"
                        f"<agent_reviews>\n{review_text}\n</agent_reviews>"
                    ),
                }
            ]
        }
    )

    return result["messages"][-1].content