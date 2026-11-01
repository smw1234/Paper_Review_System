"""
提取文本-按章节标题拆分-按审稿职责把相关的章节组合-返回字典
"""
import re
from io import BytesIO
from pypdf import PdfReader


def extract_pdf_text(pdf_bytes: bytes) -> str:
    """从 PDF 二进制内容中提取各页文本，并合并成全文。"""
    # 检查是否收到空文件
    if not pdf_bytes:
        raise ValueError("PDF 文件为空")

    # 从内存中的 PDF 字节创建读取器，不必先保存到磁盘
    reader = PdfReader(BytesIO(pdf_bytes))
    # 暂不处理加密 PDF
    if reader.is_encrypted:
        raise ValueError("PDF 已加密，当前无法读取")
    page_texts = []
    # 逐页提取文本，保留页与页之间的分隔
    for page in reader.pages:
        text = page.extract_text() or ""
        text = text.strip()

        if text:
            page_texts.append(text)

    # 合并页面文本，供后续章节划分和 agent 审稿使用
    full_text = "\n\n".join(page_texts)

    # 没有文本通常表示 PDF 是扫描图片，或文件没有可提取的文本层
    if not full_text:
        raise ValueError("没有从 PDF 中提取到文本，扫描版 PDF 后续需要 OCR")

    return full_text


# 不同职责包含的标题名
SECTION_ALIASES = {
    "problem_contribution": [
        "abstract", "摘要",
        "introduction", "引言",
        "background", "研究背景",
        "related work", "related studies", "相关工作", "相关研究",
    ],
    "method": [
        "method", "methods", "methodology",
        "approach", "our approach",
        "design", "system design", "implementation",
        "方法", "研究方法", "方法设计", "系统设计", "实现",
    ],
    "evaluation": [
        "evaluation", "experiments", "experimental evaluation",
        "results", "case study", "user study",
        "实验结果与分析", "实验评估", "评估", "结果", "案例研究", "用户研究",
    ],
    "validity": [
        "threats to validity", "limitations",
        "validity threats", "研究局限", "有效性威胁", "局限性",
    ],
    "conclusion": [
        "conclusion", "conclusions", "future work",
        "总结", "结论", "未来工作",
    ],
}

def classify_section_heading(line: str) -> str | None:
    """识别章节标题，并返回（职责，标题同行的正文)；非标题返回 None。"""

    heading = line.strip()

    for role, aliases in SECTION_ALIASES.items():
        # 先匹配较长标题，避免短标题抢先匹配
        for alias in sorted(aliases, key=len, reverse=True):
            pattern = (
                    r"^\s*(?:(?:\d+(?:\.\d+)*|[IVXLCDM]+)\.?\s+)?"
                    + re.escape(alias)
                    + r"\s*(?:[—–\-:：.]\s*)?(.*)$"
            )

            match = re.match(pattern, heading, flags=re.IGNORECASE)
            if match:
                # 摘要—正文 这种格式下，正文会在这里返回
                return role, match.group(1).strip()

    return None

def split_paper_sections(text: str) -> dict[str, str]:
    """按章节标题切分全文，并将章节内容按审稿职责归类。"""

    sections = {}
    current_role = "other"
    sections[current_role] = []

    for line in text.splitlines():
        heading_result  = classify_section_heading(line)

        if heading_result  is not None:
            current_role, inline_text = heading_result
            sections.setdefault(current_role, [])

            if inline_text:
                sections[current_role].append(inline_text)
            continue

        sections[current_role].append(line)

    # 把每个职责对应的多行文本合并成一个字符串。
    result = {}
    for role, lines in sections.items():
        content = "\n".join(lines).strip()
        if content:
            result[role] = content

    return result

def parse_and_group_pdf(pdf_bytes: bytes) -> dict[str, str]:
    """提取 PDF 文本，并按审稿职责分组返回的入口。"""
    paper_text = extract_pdf_text(pdf_bytes)
    for line in paper_text.splitlines():
        if any(word in line.lower() for word in ["conclusion", "结论", "总结", "展望"]):
            print(repr(line))
    return split_paper_sections(paper_text)
