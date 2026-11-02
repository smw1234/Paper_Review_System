from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from agents.problem_contribution_agent import problem_contribution_agent
from agents.method_agent import method_agent
from agents.evaluation_agent import evaluation_agent
from agents.validity_agent import validity_agent
from agents.conclusion_agent import conclusion_agent
from agents.report_agent import report_agent

class ReviewState(TypedDict, total=False):
    """审稿总图共享的状态。

    total=False 表示图运行过程中可以逐步填入这些字段。
    开始时只需要传入 paper_sections；
    各 agent 运行后，再分别写入自己的审稿意见。
    """

    # pdf_parser 返回的按职责分组的章节文本
    paper_sections: dict[str, str]

    # 五个审稿 agent 的输出
    problem_review: dict
    method_review: dict
    evaluation_review: dict
    validity_review: dict
    conclusion_review: dict

    average_score: float | None
    scored_dimension_count: int
    # 最终汇总报告
    review_report: dict

def problem_review_node(state: ReviewState) -> dict:
    """审阅问题定义、研究动机和贡献。"""
    sections = state["paper_sections"]

    # 如果解析器没有识别到对应章节，就传空字符串。
    # agent 函数负责返回“没有相关内容”的提示。
    review = problem_contribution_agent(
        sections.get("problem_contribution", "")
    )

    return {"problem_review": review}

def method_review_node(state: ReviewState) -> dict:
    """审阅研究方法、设计和实现。"""
    sections = state["paper_sections"]

    review = method_agent(
        sections.get("method", "")
    )

    return {"method_review": review}


def evaluation_review_node(state: ReviewState) -> dict:
    """审阅实验、评估过程和结果。"""
    sections = state["paper_sections"]

    review = evaluation_agent(
        sections.get("evaluation", "")
    )

    return {"evaluation_review": review}

def validity_review_node(state: ReviewState) -> dict:
    """审阅有效性威胁和研究局限。"""
    sections = state["paper_sections"]

    review = validity_agent(
        sections.get("validity", "")
    )

    return {"validity_review": review}


def conclusion_review_node(state: ReviewState) -> dict:
    """审阅结论是否受到论文结果支持。"""
    sections = state["paper_sections"]

    review = conclusion_agent(
        sections.get("conclusion", "")
    )

    return {"conclusion_review": review}

def calculate_average_score(reviews: list[dict]) -> tuple[float | None, int]:
    """计算有效评分的平均分，score 为 null 的项不参与计算。返回 (平均分, 参与计算的评分数量)。"""

    valid_scores = []

    for review in reviews:
        score = review.get("score")

        # 只接收 1–5 的整数评分；无法评分或无效评分不参与平均。
        if isinstance(score, int) and not isinstance(score, bool) and 1 <= score <= 5:
            valid_scores.append(score)

    if not valid_scores:
        return None, 0
    #四舍五入保留两位小数
    average = round(sum(valid_scores) / len(valid_scores), 2)
    return average, len(valid_scores)

def report_node(state: ReviewState) -> dict:
    """收集五份审稿意见，计算平均分，并调用汇总 agent 生成最终报告。"""

    reviews = [
        state.get("problem_review", {}),
        state.get("method_review", {}),
        state.get("evaluation_review", {}),
        state.get("validity_review", {}),
        state.get("conclusion_review", {}),
    ]

    average_score, scored_dimension_count = calculate_average_score(reviews)

    report = report_agent(
        problem_review=state.get("problem_review", {}),
        method_review=state.get("method_review", {}),
        evaluation_review=state.get("evaluation_review", {}),
        validity_review=state.get("validity_review", {}),
        conclusion_review=state.get("conclusion_review", {}),
        average_score=average_score,
        scored_dimension_count=scored_dimension_count,
    )
    return {
        "average_score": average_score,
        "scored_dimension_count": scored_dimension_count,
        "review_report": report,
    }

def build_review_graph():
    """创建并编译完整的论文审稿工作流。"""

    # 创建工作流构建器，并告诉它所有节点共享 ReviewState。
    builder = StateGraph(ReviewState)

    # 注册节点：名称是图中的节点名，后面是执行函数。
    builder.add_node("problem_review", problem_review_node)
    builder.add_node("method_review", method_review_node)
    builder.add_node("evaluation_review", evaluation_review_node)
    builder.add_node("validity_review", validity_review_node)
    builder.add_node("conclusion_review", conclusion_review_node)
    builder.add_node("report", report_node)

    # 开始后，同时启动五个互不依赖的审稿节点。
    builder.add_edge(START, "problem_review")
    builder.add_edge(START, "method_review")
    builder.add_edge(START, "evaluation_review")
    builder.add_edge(START, "validity_review")
    builder.add_edge(START, "conclusion_review")

    # 这条边表示：等五个审稿节点都完成后，再运行汇总节点。
    builder.add_edge(
        [
            "problem_review",
            "method_review",
            "evaluation_review",
            "validity_review",
            "conclusion_review",
        ],
        "report",
    )

    # 汇总完成后结束整张图。
    builder.add_edge("report", END)

    # compile() 将流程定义编译成可调用的图。
    return builder.compile()
