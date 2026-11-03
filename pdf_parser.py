"""
PDF 解析流程：

1. 逐页提取论文文字并保留页码。
2. 根据章节标题，把文字按审稿职责分组。
3. 记录每个职责涉及的页码。
4. 在这些页中寻找图片、图注或较多矢量绘图。
5. 把找到的页面渲染成压缩图片，附到对应职责下。
"""
import re
from io import BytesIO
from pypdf import PdfReader
import base64
import pymupdf
from PIL import Image
import logging



#添加识别图表功能
#用图表标题筛选页面。
# 例如：Figure 1、Fig. 2、Table 3、图 1、表 2。
FIGURE_CAPTION_PATTERN = re.compile(
    r"\b(?:fig(?:ure)?|table)\.?\s*\d+\b|(?:图|表)\s*[0-9一二三四五六七八九十]+",
    flags=re.IGNORECASE,
)
# 第一版限制送给视觉模型的页面数，后续可再调整。
MAX_VISUAL_PAGES_PER_SECTION = 8

# 用常见数学符号筛选可能的公式行。
# 这是初步筛选，像“x = 1”这样的短表达式也可能被识别为候选。
FORMULA_PATTERN = re.compile(
    r"[=<>≤≥≠≈∑∫√±×÷∈∉∀∃∂∇∏∞∝∧∨]"
    r"|\\(?:frac|sum|int|sqrt|alpha|beta|theta|lambda|cdot|times)\b"
)

logger = logging.getLogger(__name__)

def extract_pdf_text(pdf_bytes: bytes) -> str:
    """逐页提取 PDF 文本，并保留从 1 开始的页码"""
    # 检查是否收到空文件
    if not pdf_bytes:
        raise ValueError("PDF 文件为空")

    # 从内存中的 PDF 字节创建读取器，不必先保存到磁盘
    reader = PdfReader(BytesIO(pdf_bytes))

    # 暂不处理加密 PDF
    if reader.is_encrypted:
        raise ValueError("PDF 已加密，当前无法读取")

    pages = []
    # 逐页提取文本， 页码从 1 开始，方便和论文页码对应。
    for page_number,page in enumerate(reader.pages,start=1):

        page_text = (page.extract_text() or "").strip()
        # 即使单页没有文字，也保留页码记录。
        pages.append({
            "page": page_number,
            "text": page_text,
        })
        # 如果整篇 PDF 都没有可提取的文本，说明可能是扫描版。
    if not any(page["text"] for page in pages):
        raise ValueError("没有从 PDF 中提取到文本，扫描版 PDF 后续需要 OCR")
    return pages


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

def classify_section_heading(line: str) -> tuple[str,str] | None:
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


def split_pages_into_sections(pages: list[dict]) -> tuple[dict[str, str], dict[str, list[int]], list[list[str]]]:
    """按章节标题把逐页文本组合成各职责的正文，并记录每页职责。"""
    # 每个职责先存放多行文字，最后再拼接。
    section_lines: dict[str, list[str]] = {
        "other": [],
    }
    # 记录某职责出现在哪些页。
    section_page_numbers: dict[str, list[int]] = {
        "other": [],
    }
    # 记录每一页可能包含哪些职责。
    # 一页跨越章节边界时，这一页可以归到多个职责。
    roles_by_page: list[list[str]] = []
    # 论文开头暂时属于 other。
    current_role = "other"

    # 逐页处理，跨页时保留上一页识别到的章节职责。
    for page in pages:
        # 记录当前页出现的职责，先加入上一页延续下来的职责。
        roles_on_page = [current_role]

        for line in page["text"].splitlines():
            heading_result = classify_section_heading(line)

            # 如果当前行是章节标题，更新后续正文的职责。
            if heading_result is not None:
                current_role, inline_text = heading_result
                if current_role not in roles_on_page:
                    roles_on_page.append(current_role)
                    # 确保该职责的文字列表已创建。
                section_lines.setdefault(current_role, [])
                section_page_numbers.setdefault(current_role, [])

                # 标题同行的正文也属于这个章节。
                if inline_text:
                    section_lines[current_role].append(inline_text)

                continue

            # 普通正文追加到当前职责。
            section_lines.setdefault(current_role, [])
            section_lines[current_role].append(line)
            # 这一页的正文属于当前职责。
            if current_role not in roles_on_page:
                roles_on_page.append(current_role)

        # 记录本页对应的职责列表。
        roles_by_page.append(roles_on_page)
        # 把当前页码登记到涉及的每个职责下。
        for role in roles_on_page:
            page_numbers = section_page_numbers.setdefault(role, [])

            if page["page"] not in page_numbers:
                page_numbers.append(page["page"])

        # 把每个职责的多行正文合并成字符串，并去掉空章节。
    sections = {}

    for role, lines in section_lines.items():
        session_text = "\n".join(lines).strip()

        if session_text:
            sections[role] = session_text

    return sections, section_page_numbers, roles_by_page

def extract_page_tables(pdf_page, page_number: int) -> list[dict]:
    """从单页 PDF 提取表格，并保留行列、页码和表格序号。"""

    # 先使用默认方式检测表格，适合有边框或线条的表格。
    try:
        table_finder = pdf_page.find_tables()
        detected_tables = table_finder.tables

        # 默认方式没找到时，再用文字位置尝试检测无边框表格。
        if not detected_tables:
            table_finder = pdf_page.find_tables(strategy="text")
            detected_tables = table_finder.tables

    except Exception:
        # 表格识别失败不应导致整份 PDF 解析失败。
        logger.exception("第 %s 页表格识别失败", page_number)
        return []

    extracted_tables = []
    seen_rows = set()

    # 遍历此页检测到的表格。
    for table_index, table in enumerate(detected_tables, start=1):
        # extract() 返回二维列表：每一项是一行，每行包含多个单元格。
        raw_rows = table.extract() or []

        # 去掉单元格两端空白，并把 None 统一成空字符串。
        rows = [
            [
                (cell or "").strip()
                for cell in row
            ]
            for row in raw_rows
        ]

        # 去掉完全空白的行。
        rows = [
            row
            for row in rows
            if any(cell for cell in row)
        ]

        # 忽略只有一格或没有内容的检测结果，减少把普通文本误当表格。
        if len(rows) < 2 or max((len(row) for row in rows), default=0) < 2:
            continue

        # 用单元格内容去重，防止不同检测策略返回同一张表。
        table_key = tuple(tuple(row) for row in rows)

        if table_key in seen_rows:
            continue

        seen_rows.add(table_key)

        # 保存结构化表格内容和定位信息。
        extracted_tables.append({
            "page": page_number,
            "table_index": table_index,
            "bbox": list(table.bbox) if table.bbox else None,
            "rows": rows,
        })

    return extracted_tables

def extract_formula_candidates(page_text: str, page_number: int) -> list[dict]:
    """从页面文本中筛出可能的公式行，保留原始文字和页码。"""

    candidates = []
    seen_lines = set()

    # 一行一行筛选，避免把整段正文都当作公式。
    for line in page_text.splitlines():
        formula_line = line.strip()

        # 太短的内容通常是孤立符号，不单独作为公式候选。
        if len(formula_line) < 5:
            continue

        # 没有数学符号或常见公式命令的行先跳过。
        if not FORMULA_PATTERN.search(formula_line):
            continue

        # 同一页重复出现的公式候选只保留一次。
        if formula_line in seen_lines:
            continue

        seen_lines.add(formula_line)

        # 保留原始文本；是否准确仍要结合该页图片核对。
        candidates.append({
            "page": page_number,
            "text": formula_line,
        })

    return candidates


def page_contains_visual_content( pdf_page,
    page_text: str,
    has_table: bool = False,
    has_formula: bool = False,
) -> bool:
    """判断页面是否可能含图、表或公式等视觉内容。"""
    # 页面文字中存在图题或表题时，认为它是视觉页面候选。
    has_caption = bool(FIGURE_CAPTION_PATTERN.search(page_text))

    # get_image_info 能发现页面中实际显示的嵌入图片。
    has_embedded_image = bool(pdf_page.get_image_info())

    # get_drawings 能发现 PDF 中的线条、矩形、曲线等矢量绘图。
    # 用数量阈值做初筛，因为普通页面也可能含少量线条或装饰。
    # 这个阈值是启发式的，后续可根据你的论文样例调高或调低。
    has_many_vector_drawings = len(pdf_page.get_drawings()) >= 20

    # 任一条件满足，就把整页渲染给对应章节的 agent。
    return has_table or has_formula or has_caption or has_embedded_image or has_many_vector_drawings

def render_page_as_base64(pdf_page) -> str:
    """把 PDF 页面渲染、缩放并压缩为 JPEG Base64 字符串。"""
    # 以 160 DPI 渲染页面，尽量保留图表中的小字。
    pixmap = pdf_page.get_pixmap(
        dpi=160,
        colorspace=pymupdf.csRGB,
        alpha=False,
    )

    # 把 PyMuPDF 的 PNG 字节交给 Pillow 处理。
    image = Image.open(BytesIO(pixmap.tobytes("png")))

    # 限制最大尺寸，减少发送给视觉模型的数据量。
    image.thumbnail(
        (1400, 1800),
        Image.Resampling.LANCZOS,
    )

    # 将图像转成 RGB，并压缩成 JPEG。
    jpeg_buffer = BytesIO()
    image.convert("RGB").save(
        jpeg_buffer,
        format="JPEG",
        quality=82,
        optimize=True,
    )

    # 模型接口使用 Base64 data URL 接收图片。
    return base64.b64encode(
        jpeg_buffer.getvalue()
    ).decode("utf-8")

def attach_visual_pages(
    pdf_bytes: bytes,
    pages: list[dict],
    roles_by_page: list[list[str]],
) ->  tuple[dict, dict, dict, dict]:
    """ 按职责提取表格、公式候选和页面图片。"""

    # 保存职责对应的页面图片。
    section_images: dict[str, list[dict]] = {}

    # 每个职责对应的结构化表格。
    section_tables: dict[str, list[dict]] = {}

    # 每个职责对应的公式候选行。
    section_formulas: dict[str, list[dict]] = {}

    # 记录因每个职责的图片数量限制而省略的页面数。
    omitted_visual_pages: dict[str, int] = {}

    # 用 PyMuPDF 打开上传的 PDF，以便检测并渲染页面。
    document = pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf",
    )

    try:
        # 同时遍历提取出的页文本、职责关系和 PDF 页面对象。
        for page_index, (page_info, page_roles) in enumerate(
                zip(pages, roles_by_page)
        ):
            # PyMuPDF 的页面索引从 0 开始。
            pdf_page = document[page_index]

            # 页码按 1 开始，和 pypdf 提取结果对应。
            page_number = page_info["page"]

            # 提取本页结构化表格。
            page_tables = extract_page_tables(
                pdf_page,
                page_number,
            )

            # 从本页提取可能的公式文本行。
            page_formulas = extract_formula_candidates(
                page_info["text"],
                page_number,
            )
            # 同一页可能属于两个章节，例如页面中途换章。
            # 因此表格和公式会分别归到这一页涉及的职责。
            for role in page_roles:
                section_tables.setdefault(role, [])
                section_formulas.setdefault(role, [])
                section_tables[role].extend(page_tables)
                section_formulas[role].extend(page_formulas)

            # 公式候选不为空时，将其所在页面也渲染，供视觉 agent 核对。
            has_table = bool(page_tables)
            has_formula = bool(page_formulas)

        # 没有视觉内容的页面无需转换成图片。
            if not page_contains_visual_content(
                pdf_page,
                page_info["text"],
                has_table=has_table,
                has_formula=has_formula,
              ):
                continue

        # 计算这张图片还能发送给哪些职责 agent。
            roles_to_attach = []

            for role in page_roles:
                images_for_role = section_images.setdefault(role, [])

                # 每个职责最多发送配置数量的页面图片。
                if len(images_for_role) < MAX_VISUAL_PAGES_PER_SECTION:
                    roles_to_attach.append(role)
                else:
                    # 超出数量限制时记录页数，后续会告诉 agent。
                    omitted_visual_pages[role] = (
                            omitted_visual_pages.get(role, 0) + 1
                    )

        # 所有相关职责都达到上限时，跳过图片渲染。
            if not roles_to_attach:
                 continue

             # 将 PDF 页面渲染并压缩成 Base64 JPEG。
            image_base64 = render_page_as_base64(pdf_page)

            # 一页跨章节时，同一页图片会附给多个相关 agent。
            for role in roles_to_attach:
                section_images[role].append({
                    "page": page_number,
                    "page_text": page_info["text"],
                    "image_base64": image_base64,
                })

    finally:
        # 无论渲染成功还是失败，都关闭 PDF 文档。
        document.close()

    return  (
        section_images,
        omitted_visual_pages,
        section_tables,
        section_formulas,
    )

def parse_and_group_pdf(pdf_bytes: bytes) -> dict:
     """解析 PDF，返回章节文字、页码、图片、表格和公式候选。"""
     # 先逐页提取文字。
     pages = extract_pdf_text(pdf_bytes)
     # 按章节标题组合正文，并记录章节页码。
     sections, section_pages, roles_by_page = split_pages_into_sections(pages)
     # 提取视觉页面、表格结构、公式候选和省略页面数量。
     (
         section_images,
         omitted_visual_pages,
         section_tables,
         section_formulas,
     ) = attach_visual_pages(
         pdf_bytes,
         pages,
         roles_by_page,
     )

     # 把所有解析结果按职责分别返回给 main 和 review_graph。
     return {
         "sections": sections,
         "section_pages": section_pages,
         "section_images": section_images,
         "section_tables": section_tables,
         "section_formulas": section_formulas,
         "omitted_visual_pages": omitted_visual_pages,
     }

