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



#添加识别图表功能
# 第一版用图表标题筛选页面。
# 例如：Figure 1、Fig. 2、Table 3、图 1、表 2。
FIGURE_CAPTION_PATTERN = re.compile(
    r"\b(?:fig(?:ure)?|table)\.?\s*\d+\b|(?:图|表)\s*[0-9一二三四五六七八九十]+",
    flags=re.IGNORECASE,
)
# 第一版限制送给视觉模型的页面数，后续可再调整。
MAX_VISUAL_PAGES_PER_SECTION = 8



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

def page_contains_visual_content(pdf_page, page_text: str) -> bool:
    """判断页面是否可能含有图片、图表、表格或矢量绘图。"""
    # 页面文字中存在图题或表题时，认为它是视觉页面候选。
    has_caption = bool(FIGURE_CAPTION_PATTERN.search(page_text))

    # get_image_info 能发现页面中实际显示的嵌入图片。
    has_embedded_image = bool(pdf_page.get_image_info())

    # get_drawings 能发现 PDF 中的线条、矩形、曲线等矢量绘图。
    # 用数量阈值做初筛，因为普通页面也可能含少量线条或装饰。
    # 这个阈值是启发式的，后续可根据你的论文样例调高或调低。
    has_many_vector_drawings = len(pdf_page.get_drawings()) >= 20

    # 任一条件满足，就把整页渲染给对应章节的 agent。
    return has_caption or has_embedded_image or has_many_vector_drawings

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
) -> tuple[dict[str, list[dict]], dict[str, int]]:
    """把视觉页面按章节职责分组并转换为 Base64 图片。"""
    # 保存职责对应的页面图片。
    section_images: dict[str, list[dict]] = {}

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

            # 没有图表标题、嵌入图片或足够矢量绘图的页面先跳过。
            if not page_contains_visual_content(
                    pdf_page,
                    page_info["text"],
            ):
                continue

            # 判断这张页面图还需要附到哪些职责下。
            roles_to_attach = []

            for role in page_roles:
                images_for_role = section_images.setdefault(role, [])

                # 每个职责最多传入指定数量的页面图。
                if len(images_for_role) < MAX_VISUAL_PAGES_PER_SECTION:
                    roles_to_attach.append(role)
                else:
                    # 超出限制的页面只记录数量，不静默忽略。
                    omitted_visual_pages[role] = (
                            omitted_visual_pages.get(role, 0) + 1
                    )

            # 如果所有相关职责都达到数量上限，就不再渲染这页。
            if not roles_to_attach:
                continue

            # 渲染页面并转成 Base64，后续交给对应 agent。
            image_base64 = render_page_as_base64(pdf_page)

            # 一页若跨越多个章节，会分别附给这些职责的 agent。
            for role in roles_to_attach:
                section_images[role].append({
                    # 保留页码，agent 可以在意见中引用位置。
                    "page": page_info["page"],

                    # 同时保留页面文本，方便文字和图表相互印证。
                    "page_text": page_info["text"],

                    # 图片内容以 Base64 形式交给视觉模型。
                    "image_base64": image_base64,
                })

    finally:
        # 无论渲染成功还是失败，都关闭 PDF 文档。
        document.close()

    return section_images, omitted_visual_pages

def parse_and_group_pdf(pdf_bytes: bytes) -> dict:
     """解析 PDF 并返回章节正文、页码和按职责分组的页面图片。"""
     # 先逐页提取文字。
     pages = extract_pdf_text(pdf_bytes)
     # 按章节标题组合正文，并记录章节页码。
     sections, section_pages, roles_by_page = split_pages_into_sections(pages)
     # 查找并渲染每个职责相关的视觉页面。
     section_images, omitted_visual_pages = attach_visual_pages(
         pdf_bytes,
         pages,
         roles_by_page,
     )
     # 返回对象包含正文和图像，后续由 review_graph 分配给 agent。
     return {
         # 例如：paper_sections["evaluation"] 是实验章节正文。
         "sections": sections,

         # 例如：section_pages["evaluation"] 是实验章节所在页码。
         "section_pages": section_pages,

         # 例如：section_images["evaluation"] 是实验章节页面图片列表。
         "section_images": section_images,

         # 记录每个职责因数量上限而未附加的页面数。
         "omitted_visual_pages": omitted_visual_pages,
     }

