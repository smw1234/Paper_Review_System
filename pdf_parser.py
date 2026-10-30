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
        "实验", "实验评估", "评估", "结果", "案例研究", "用户研究",
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
    """识别章节标题，并返回对应的审稿职责；普通正文返回 None。"""

    heading = line.strip()

    # 去掉章节编号，例如“2 Methodology”或“3.1 Evaluation”。
    numbered_heading = re.match(r"^\d+(?:\.\d+)*\.?\s+(.+)$", heading)
    if numbered_heading:
        heading = numbered_heading.group(1).strip()
        is_numbered = True
    else:
        is_numbered = False

    normalized_heading = heading.lower()

    # 根据标题别名查找对应职责。
    for role, aliases in SECTION_ALIASES.items():
        normalized_aliases = [alias.lower() for alias in aliases]
        if normalized_heading in normalized_aliases:
            return role

    # 有编号但暂时不认识的标题，先归到 other，避免丢失该节内容。
    if is_numbered:
        return "other"

    return None

def split_paper_sections(text: str) -> dict[str, str]:
    """按章节标题切分全文，并将章节内容按审稿职责归类。"""

    sections = {}
    current_role = "other"
    sections[current_role] = []

    for line in text.splitlines():
        role = classify_section_heading(line)

        if role is not None:
            current_role = role
            sections.setdefault(current_role, [])
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
    return split_paper_sections(paper_text)
