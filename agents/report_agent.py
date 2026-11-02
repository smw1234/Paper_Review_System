import os
import json
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI


llm = ChatOpenAI(
    model=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url=os.environ.get(
        "DEEPSEEK_BASE_URL",
        "https://api.deepseek.com",
    ),
    model_kwargs={
        "extra_body": {
            "thinking": {
                "type": "disabled",
            }
        }
    },
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

评分汇总要求：
1. 汇总各审稿 agent 提供的评分与依据，不要自行修改或补造分数。
2. “无法评分”的维度不参与平均分计算，并说明该维度未参与计算。
3. 总分为所有有效维度分数的算术平均值，保留两位小数。
4. 同时列出参与平均分计算的维度及各自分数。
5. 如果没有任何有效分数，写“暂无可计算的总分”。
6. 合并重复意见，但保留实质不同的问题。
7. 如果 agent 的意见有冲突，指出冲突；不要擅自编造结论来消除冲突。
8. 只依据各 agent 提供的意见和论文依据生成报告。

请按以下 Markdown 结构输出：

# 审稿报告

## 论文优点
## 主要问题
## 次要问题与修改建议
## 需要作者补充说明的内容
## 总体评价
""",
)


def report_agent(problem_review: dict,method_review: dict,evaluation_review: dict,validity_review: dict,conclusion_review: dict, average_score: float | None,
    scored_dimension_count: int,) -> str:
    """汇总各审稿 agent 的意见，生成结构化审稿报告。"""

    reviews = {
        "问题与贡献审稿意见": problem_review,
        "研究方法审稿意见": method_review,
        "实验与评估审稿意见": evaluation_review,
        "有效性与局限审稿意见": validity_review,
        "结论审稿意见": conclusion_review,
    }
    review_data = json.dumps(
        {
            "reviews": reviews,
            "average_score": average_score,
            "scored_dimension_count": scored_dimension_count,
        },
        ensure_ascii=False,
        indent=2,
    )

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content":  (
                        "请根据以下结构化审稿意见生成最终报告。\n"
                        "平均分由程序计算完成，请直接使用给定分数，"
                        "不要重新计算、修改或补造分数。\n"
                        "score 为 null 的职责不参与平均分，"
                        "请注明实际参与评分的维度数量。\n"
                        "只总结输入中有依据的内容；信息不足时明确说明，"
                        "不要编造论文事实。\n\n"
                        f"<review_data>\n{review_data}\n</review_data>"
                    ),
                }
            ]
        }
    )

    summary = result["messages"][-1].content

    # 将五个 agent 的列表字段合并，供前端分区展示。
    strengths = [
        item
        for review in reviews.values()
        for item in review.get("strengths", [])
    ]

    issues = [
        item
        for review in reviews.values()
        for item in review.get("issues", [])
    ]

    suggestions = [
        item
        for review in reviews.values()
        for item in review.get("suggestions", [])
    ]

    # 返回结构与前端 app.js 读取的字段对应。
    return {
        "title": "论文审稿报告",
        "average_score": average_score,
        "scored_dimension_count": scored_dimension_count,
        "summary": summary,
        "strengths": strengths,
        "issues": issues,
        "suggestions": suggestions,
        "reviews": reviews,
    }