import json
import logging
import operator
from typing import Annotated
from typing import TypedDict
from typing import Any
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
    # 保存审稿节点的错误信息。
    # operator.add 会把并行节点各自返回的错误列表合并起来，
    # 避免多个节点同时写同一个普通字段造成状态更新冲突。
    errors: Annotated[list[str], operator.add]

    # 每个职责对应的 PDF 页码。
    section_pages: dict[str, list[int]]

    # 每个职责对应的页面图片列表。
    # 每个图片项含 page、page_text、image_base64。
    section_images: dict[str, list[dict[str, Any]]]

    # 每个职责因图片数量上限而省略的页面数。
    omitted_visual_pages: dict[str, int]
    # 各职责识别到的结构化表格。
    section_tables: dict[str, list[dict[str, Any]]]

    # 各职责识别到的公式候选文字。
    section_formulas: dict[str, list[dict[str, Any]]]

logger = logging.getLogger(__name__)

def run_review_agent(agent_function,state: ReviewState,
    section_name: str,
    result_key: str,
    agent_name: str,
) -> dict:
    """统一给章节 agent 传正文、表格、公式候选和页面图片。"""
    try:
        #取出正文
        sections = state.get("paper_sections",{})
        section_text = sections.get(section_name, "")
        # 取出当前职责对应的页面图片。
        all_section_images = state.get("section_images", {})
        visual_pages = all_section_images.get(section_name, [])
        # 取出当前职责的结构化表格数据。
        all_section_tables = state.get("section_tables", {})
        section_tables = all_section_tables.get(section_name, [])

        # 取出当前职责的公式候选文本。
        all_section_formulas = state.get("section_formulas", {})
        section_formulas = all_section_formulas.get(section_name, [])
        # 取出当前职责因图片数量限制而省略的页数。
        omitted_pages_by_section = state.get(
            "omitted_visual_pages",
            {},
        )
        omitted_visual_pages = omitted_pages_by_section.get(
            section_name,
            0,
        )
        # 将表格以可读 JSON 格式追加到章节文本。
        if section_tables:
            section_text += (
                    "\n\n<structured_tables>\n"
                    + json.dumps(
                section_tables,
                ensure_ascii=False,
                indent=2,
            )
                    + "\n</structured_tables>"
            )
            # 将公式候选及其页码追加到章节文本。
        if section_formulas:
            section_text += (
                    "\n\n<formula_candidates>\n"
                    + json.dumps(
                section_formulas,
                ensure_ascii=False,
                indent=2,
            )
                    + "\n</formula_candidates>"
            )

        # 调用 agent；现有 agent 函数接收正文和页面图片。
        # 表格和公式候选已经合并进 section_text。
        review = agent_function(
            section_text,
            visual_pages=visual_pages,
            omitted_visual_pages=omitted_visual_pages,
        )
        # 把当前 agent 的结构化审稿结果写入图状态。
        return {result_key: review}

    except Exception as exc:
        logger.exception("%s 调用失败", agent_name)
        return {
            # 每个失败节点只返回自己的错误。
            # LangGraph 会通过 operator.add 合并并行节点返回的列表。
            "errors": [
                f"{agent_name}调用失败：{type(exc).__name__}: {exc}"
            ]
        }

def problem_review_node(state: ReviewState) -> dict:
    """审阅问题定义、研究动机和贡献。"""
    return run_review_agent(
        problem_contribution_agent,
        state,
        "problem_contribution",
        "problem_review",
        "问题与贡献审稿 agent",
    )

def method_review_node(state: ReviewState) -> dict:
    """审阅研究方法、设计和实现。"""
    return run_review_agent(
        method_agent,
        state,
        "method",
        "method_review",
        "研究方法审稿 agent",
    )


def evaluation_review_node(state: ReviewState) -> dict:
    """审阅实验、评估过程和结果。"""
    return run_review_agent(
        evaluation_agent,
        state,
        "evaluation",
        "evaluation_review",
        "实验与评估审稿 agent",
    )


def validity_review_node(state: ReviewState) -> dict:
    """审阅有效性威胁和研究局限。"""
    return run_review_agent(
        validity_agent,
        state,
        "validity",
        "validity_review",
        "有效性与局限审稿 agent",
    )


def conclusion_review_node(state: ReviewState) -> dict:
    """审阅结论是否受到论文结果支持。"""
    return run_review_agent(
        conclusion_agent,
        state,
        "conclusion",
        "conclusion_review",
        "结论审稿 agent",
    )

def check_reviews_node(state: ReviewState) -> dict:
    """五个审稿节点完成后的汇合点；只负责让图检查 error 状态。"""
    return {}

def route_after_reviews(state: ReviewState) -> str:
    """检查并行审稿节点是否有失败；有失败就跳过汇总。"""
    if state.get("errors"):
        return "end"
    return "report"


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
    builder.add_node("check_reviews", check_reviews_node)
    builder.add_node("report", report_node)

    # 开始后，同时启动五个互不依赖的审稿节点。
    builder.add_edge(START, "problem_review")
    builder.add_edge(START, "method_review")
    builder.add_edge(START, "evaluation_review")
    builder.add_edge(START, "validity_review")
    builder.add_edge(START, "conclusion_review")

    # 五个节点都完成后汇合到同一个检查节点
    builder.add_edge("problem_review", "check_reviews")
    builder.add_edge("method_review", "check_reviews")
    builder.add_edge("evaluation_review", "check_reviews")
    builder.add_edge("validity_review", "check_reviews")
    builder.add_edge("conclusion_review", "check_reviews")

    # 这条边表示：等五个审稿节点都完成后，再运行汇总节点。
    builder.add_conditional_edges(
        "check_reviews",
        route_after_reviews,
    {
        "report": "report",
        "end": END,
    },
    )

    # 汇总完成后结束整张图。
    builder.add_edge("report", END)

    # compile() 将流程定义编译成可调用的图。
    return builder.compile()
