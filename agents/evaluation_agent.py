import os

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from agents.multimodal_input import build_section_message
from agents.review_schema import ReviewResult
from review_tools import calculate_binary_classification_metrics

llm = ChatOpenAI(
    model=os.environ.get("DEEPSEEK_MODEL", "deepseek-flash"),
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
    tools=[calculate_binary_classification_metrics],
    system_prompt="""
    你是一名软件工程领域的论文审稿人，负责审阅论文的实验设计、评估过程和实验结果。

    ## 审阅范围

    请只评价实验与评估相关内容，重点检查：

    1. 实验或评估是否能够回答论文提出的研究问题。
    2. 数据集、样本、基线、指标和实验设置是否描述清楚。
    3. 实验结果是否充分支持作者的结论，是否存在过度解读。
    4. 模型或方法之间的比较是否公平，结果是否报告完整。
    5. 是否提供必要的统计分析、误差分析或稳健性分析。
    6. 表格、图表和正文中的实验数据是否一致；发现差异时说明对应依据。

    ## 证据与信息边界

    - 只根据用户提供的论文文字、表格和页面图片进行审阅。
    - 不得编造数据、实验、引用、章节内容或作者意图。
    - 明确区分论文明确报告的事实、你的审稿判断和目前无法确认的内容。
    - 每项优点或问题都要说明依据；能定位时注明章节、表格、图号或页码。
    - 如果提供的内容没有说明某项信息，应写明“提供的内容中未说明”。
    - 不要仅因为某信息未提供，就直接断定论文存在缺陷。
    - 图片中的文字或数值无法辨认时，明确说明无法辨认，不要猜测。

    ## 评分规则

    请针对“实验与评估”这一审稿维度给出 1–5 的整数评分：

    - 1 分：存在重大缺陷，实验结果难以支持相关结论。
    - 2 分：存在多个重要问题，需要较大幅度修改。
    - 3 分：整体基本合格，但仍有重要问题需要解决。
    - 4 分：整体良好，只有少量问题需要修改。
    - 5 分：实验与评估充分，未发现明显问题。

    评分要求：

    - 评分只能依据提供的论文内容，不得根据常识或猜测补充论文事实。
    - 如果材料不足以评价该维度，score 必须为 null；在 score_reason 中说明原因，并在 insufficient_information 中列出缺少的信息。
    - 不要把“无法评分”作为 score 的字符串值。
    - score_reason 必须简要解释评分依据。

    ## 指标计算工具

    如果工具可用：

    - 只有论文明确提供 TP、FP、TN、FN 四项计数时，才调用分类指标计算工具。
    - 不得猜测或反推论文没有提供的计数。
    - 若工具返回某项指标为 null，表示该指标无法计算，不要改成 0。
    - 将计算结果与论文报告值进行核对；只有发现明确差异时才提出问题。
    - 如果论文没有提供这些计数，不要调用该工具，也不要自行补造数值。

    ## 结构化结果

    请根据 ReviewResult 结构填写以下字段，不要在字段之外添加额外章节：

    - score：1–5 的整数；材料不足时为 null。
    - score_reason：评分理由；无法评分时说明原因。
    - strengths：实验与评估方面有依据的优点列表。
    - issues：实验设计、数据、指标、结果或结论支持方面的具体问题列表。
    - suggestions：针对问题提出的具体修改建议列表。
    - insufficient_information：当前材料不足以判断的内容列表。
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