"""agent tools"""
from langchain.tools import tool


@tool
def calculate_binary_classification_metrics(
    tp: int,
    fp: int,
    tn: int,
    fn: int,
) -> dict:
    """根据明确提供的混淆矩阵计数计算二分类指标(准确率 精确率 召回率 特异度 F1分数）。

    参数必须是论文中明确报告的计数：
    tp：真正例
    fp：假正例
    tn：真负例
    fn：假负例

    返回的指标是 0 到 1 之间的比例。
    分母为 0 时，该指标返回 None。
    """

    # 拒绝负数，防止把百分比或其他数值误当作计数。
    counts = (tp, fp, tn, fn)#组成元组
    if any(value < 0 for value in counts):
        return {"error": "TP、FP、TN、FN 必须是非负整数。"}

    # 计算每项指标的分母。
    total = tp + fp + tn + fn
    precision_denominator = tp + fp
    recall_denominator = tp + fn
    specificity_denominator = tn + fp
    f1_denominator = 2 * tp + fp + fn

    # 计算各项指标；分母为 0 时返回 None，不擅自当成 0。
    accuracy = (tp + tn) / total if total else None
    precision = (
        tp / precision_denominator
        if precision_denominator
        else None
    )
    recall = (
        tp / recall_denominator
        if recall_denominator
        else None
    )
    specificity = (
        tn / specificity_denominator
        if specificity_denominator
        else None
    )
    f1 = (
        2 * tp / f1_denominator
        if f1_denominator
        else None
    )

    # 四舍五入到小数点后四位，便于报告展示。
    def round_or_none(value):
        return round(value, 4) if value is not None else None

    return {
        "input_counts": {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        },
        "metrics": {
            "accuracy": round_or_none(accuracy),
            "precision": round_or_none(precision),
            "recall": round_or_none(recall),
            "specificity": round_or_none(specificity),
            "f1": round_or_none(f1),
        },
        "note": "指标为 0 到 1 之间的比例；None 表示该指标无法计算。",
    }

import json
import os
import re
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from langchain.tools import tool


@tool
def search_related_papers(query: str, top_k: int = 5) -> str:
    """搜索与论文问题、方法或贡献相近的学术论文。

    query 应使用简短的研究主题和技术关键词。
    返回的论文是候选相关工作，不代表已经证明原论文缺乏创新性。
    """

    # 检查查询内容，避免空查询或过长查询。
    # 去掉首尾空格，并把换行、制表符等空白统一成空格。
    query = re.sub(r"\s+", " ", query).strip()
    # 避免把整段论文内容作为查询词发送给外部服务。
    # 查询最多允许 12 个词，且总长度不能超过 160 个字符。
    query_words = query.split()
    if len(query) > 160 or len(query_words) > 12:
        return json.dumps(
            {
                "status": "error",
                "message": (
                    "检索词过长。请改用 3–8 个概括性主题或方法关键词，"
                    "不要复制论文标题、摘要或正文。"
                ),
            },
            ensure_ascii=False,
        )

    # 只保留关键词中常见的字母、数字、中文和少量连接符。
    # 去掉引号等标点，避免将完整句子作为检索表达式。
    query = re.sub(r"[^\w\s+#.-]", " ", query, flags=re.UNICODE)
    query = re.sub(r"\s+", " ", query).strip()

    # 查询词太短时不向 Semantic Scholar 发送请求。
    if len(query) < 3:
        return json.dumps(
            {"status": "error", "message": "检索词至少需要 3 个字符。"},
            ensure_ascii=False,
        )

    # 限制检索数量，控制返回给 agent 的论文数量。
    top_k = max(1, min(top_k, 8))

    # 将关键词作为查询参数发送；不要在这里传入论文正文。
    params = urlencode({
        "query": query,
        "limit": top_k,
        "fields": "title,year,authors,venue,abstract,url,externalIds",
    })

    # Semantic Scholar Academic Graph 的论文相关性搜索接口。
    api_url = (
        "https://api.semanticscholar.org/graph/v1/paper/search?"
        + params
    )

    # API key 可选；若设置了环境变量，就放入官方要求的请求头。
    headers = {
        "User-Agent": "PaperReviewSystem/1.0",
        "Accept": "application/json",
    }
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key

    # 发出搜索请求，并设置超时，避免检索一直卡住审稿流程。
    request = Request(api_url, headers=headers)

    try:
        with urlopen(request, timeout=20) as response:
            search_result = json.loads(response.read().decode("utf-8"))

    except HTTPError as exc:
        # 对限流单独说明，方便 agent 和用户区分“没搜到”和“服务限制”。
        if exc.code == 429:
            message = "Semantic Scholar 当前限制了请求频率，请稍后重试。"
        else:
            message = f"Semantic Scholar 返回 HTTP {exc.code}。"

        return json.dumps(
            {"status": "error", "message": message},
            ensure_ascii=False,
        )

    except (URLError, TimeoutError):
        # 网络或超时错误不应被误解成“没有相关工作”。
        return json.dumps(
            {
                "status": "error",
                "message": "暂时无法连接 Semantic Scholar，检索未完成。",
            },
            ensure_ascii=False,
        )

    # 整理返回结果，只保留审稿时需要的信息。
    papers = []

    for paper in search_result.get("data", []):
        external_ids = paper.get("externalIds") or {}
        paper_id = paper.get("paperId")

        # 如果 API 没返回论文链接，用论文 ID 构造 Semantic Scholar 页面链接。
        paper_url = paper.get("url")
        if not paper_url and paper_id:
            paper_url = f"https://www.semanticscholar.org/paper/{paper_id}"

        # 作者只保留名字，限制数量以控制工具返回长度。
        authors = [
            author.get("name", "")
            for author in (paper.get("authors") or [])[:6]
            if author.get("name")
        ]

        # 摘要可能很长；截断后仍保留比较主题所需的主要内容。
        abstract = (paper.get("abstract") or "").strip()
        if len(abstract) > 1500:
            abstract = abstract[:1500] + "…"

        papers.append({
            "title": paper.get("title"),
            "authors": authors,
            "year": paper.get("year"),
            "venue": paper.get("venue"),
            "doi": external_ids.get("DOI"),
            "url": paper_url,
            "abstract": abstract or None,
        })

    # 明确标注检索结果的边界，避免把空结果当作“证明创新”。
    return json.dumps(
        {
            "status": "success",
            "source": "Semantic Scholar",
            "query": query,
            "total_matches": search_result.get("total"),
            "papers": papers,
            "interpretation_limit": (
                "这些是检索到的候选相关论文。无结果不证明论文具有创新性；"
                "只有标题或摘要时，也不能据此断定方法细节完全相同。"
            ),
        },
        ensure_ascii=False,
        indent=2,
    )