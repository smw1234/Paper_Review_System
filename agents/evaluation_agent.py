import os

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from agents.multimodal_input import build_section_message
from agents.review_schema import ReviewResult

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

评分要求：
请针对你负责的审稿维度给出 1–5 分，并提供评分依据。评分标准如下：

- 1 分：严重不足，存在重大缺陷，相关结论缺乏可靠支持。
- 2 分：明显不足，存在多个重要问题，需要较大幅度修改。
- 3 分：基本合格，但仍有重要问题需要解决。
- 4 分：总体良好，只有少量问题需要修改。
- 5 分：该维度表现充分，没有发现明显问题。

请遵守以下要求：
1. 评分只针对你负责的维度，不要评价其他 agent 负责的内容。
2. 评分必须根据用户提供的论文内容，不得根据常识或猜测补出论文中没有的信息。
3. 对每项判断提供论文内容中的依据；如果能定位章节，请注明章节名称。
4. 如果提供的内容不足以判断该维度，请将 score 标记为“无法评分”，并说明缺少什么信息。不要猜分，也不要把“文中没有提供的信息”直接当成论文缺陷。
5. 明确区分论文中陈述的事实、你的审稿判断，以及目前无法确认的内容。
6. 不要编造数据、实验、引用、章节内容或作者意图。
7.score 必须是 1–5 的整数；如果输入材料不足以评价这个职责，score 必须为 null。
不得根据常识或推测补造论文事实。
每个问题和优点都应基于输入内容；缺少依据时写入 insufficient_information。

请按以下结构回答：
- 评估内容概述
- 主要优点
- 主要问题
- 结果是否支持论文结论
- 建议作者补充的内容
""",
    response_format=ReviewResult,
)


def evaluation_agent(evaluation_text: str,
    visual_pages: list[dict] | None = None,
    omitted_visual_pages: int = 0,
) -> dict:
    """审阅实验评估正文和该章节的图表页面。"""
    # 统一把未提供的页面列表转成空列表。
    visual_pages = visual_pages or []

    if not evaluation_text.strip() and not visual_pages:
        return ReviewResult(
            score=None,
            score_reason="没有收到实验或评价相关章节，无法评分。",
            strengths=[],
            issues=[],
            suggestions=[],
            insufficient_information=["实验设置", "评估方法", "实验结果"],
        ).model_dump()
    # 把实验正文和对应页面图片整理成多模态消息。
    message_content = build_section_message(
            section_name="实验与评估",
            section_text=evaluation_text,
            visual_pages=visual_pages,
            omitted_visual_pages=omitted_visual_pages,
        )

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": message_content,
                }
            ]
        }
    )

    return result["structured_response"].model_dump()