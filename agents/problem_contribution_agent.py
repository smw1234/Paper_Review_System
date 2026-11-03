import os
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from agents.review_schema import ReviewResult

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

system_prompt = """
你是一名软件工程领域的论文审稿人。
请根据输入的论文内容，审阅研究问题、研究动机和论文贡献。

请遵守以下要求：
1. 只根据输入内容判断，不要编造信息。
2. 分析研究问题是否清楚、重要，动机是否充分。
3. 分析作者声称的贡献是否清楚，以及论文是否说明了与相关工作的区别。
4. 如果当前内容不足以判断，请明确写出“无法根据提供的章节判断”。
5. 说明判断依据，不要只给结论。

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
- 问题与动机
- 贡献清晰度
- 与相关工作的区别
- 主要优点
- 主要疑问
- 尚无法判断的内容
"""

# 创建“问题与贡献审稿” agent
agent = create_agent(
    model=llm,
    tools=[],
    system_prompt= system_prompt,
    response_format=ReviewResult,)


def problem_contribution_agent(paper_text:str)->dict:
    """审阅论文的问题定义、研究动机和贡献。"""
    if not paper_text.strip():
        return  ReviewResult(
            score=None,
            score_reason="未能识别相关章节，无法评分。",
            strengths=[],
            issues=[],
            suggestions=[],
            insufficient_information=["摘要、引言或相关工作章节"],
        ).model_dump()

    response=agent.invoke(
        {"messages": [
            {
                "role": "user",
                "content":(
                        "只根据以下论文内容进行审阅。"
                        "不要补造未提供的信息；判断所需信息缺失时，"
                        "将 score 设为 null，并说明缺少什么。\n\n"
                        f"<paper_sections>\n{paper_text}\n</paper_sections>"
                    ),
            }
        ]}
    )
    # create_agent 按 ReviewResult 格式返回结构化结果。
    return response["structured_response"].model_dump()











