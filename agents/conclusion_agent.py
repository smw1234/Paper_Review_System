import os

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from agents.review_schema import ReviewResult
from agents.multimodal_input import build_section_message

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
- 结论概述
- 结论与结果的一致性
- 结论范围是否恰当
- 主要优点
- 主要问题
- 建议作者修改或补充的内容
""",
    response_format=ReviewResult,
)


def conclusion_agent( conclusion_text: str,
    visual_pages: list[dict] | None = None,
    omitted_visual_pages: int = 0,
) -> dict:
    """审阅结论正文和结论章节所在页面的图片。"""

    if not conclusion_text.strip() and not visual_pages:
        return ReviewResult(
            score=None,
            score_reason="没有收到论文结论相关章节，无法评分。",
            strengths=[],
            issues=[],
            suggestions=[],
            insufficient_information=["结论", "未来工作"],
        ).model_dump()
    # 构造含结论正文与页面图像的消息。
    message_content = build_section_message(
        section_name="结论",
        section_text=conclusion_text,
        visual_pages=visual_pages,
        omitted_visual_pages=omitted_visual_pages,
    )
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content":message_content,
                }
            ]
        }
    )

    return result["structured_response"].model_dump()