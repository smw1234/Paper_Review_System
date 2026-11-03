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
    max_retries=3,
    timeout=60,
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
你是一名软件工程领域的资深论文审稿人，负责撰写简短的总体评价。

请根据各审稿 agent 提供的结构化意见，写一段总体评价。

要求：
1. 只总结论文整体表现、最重要的优点和主要风险。
2. 不要再次罗列详细优点、问题、修改建议或各 agent 的意见。
   这些内容会在报告的其他字段中单独展示。
3. 不要输出标题、评分表、Markdown 列表或 JSON，只输出一段简洁的正文。
4. 只依据输入中的意见，不添加或推测论文事实。
5. 不要计算、修改或重复输出平均分。
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
                        "请根据以下审稿意见，只写一段简洁的总体评价。"
                        "不要重复列出详细优点、问题、建议或分数。"
                        "只依据输入内容，不要编造论文事实。\n\n"
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
    score_details = {
        name: {
            "score": review.get("score"),
            "score_reason": review.get("score_reason", ""),
        }
        for name, review in reviews.items()
    }
    # 返回结构与前端 app.js 读取的字段对应。
    return {
        "title": "论文审稿报告",
        "average_score": average_score,
        "scored_dimension_count": scored_dimension_count,
        "summary": summary,
        "strengths": strengths,
        "issues": issues,
        "suggestions": suggestions,
        "reviews":  score_details,
    }