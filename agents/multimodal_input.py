"""构造同时包含章节文字和页面图片的 agent 输入。"""

def build_section_message(
    section_name: str,
    section_text: str,
    visual_pages: list[dict] | None = None,
    omitted_visual_pages: int = 0,
) -> list[dict]:
    """生成可以发送给视觉模型的 user 消息内容。"""

    # 没传图片时使用空列表，避免后面遍历 None。
    visual_pages = visual_pages or []

    # 给模型的文字说明，明确审阅职责和证据边界。
    user_text = (
        f"请审阅以下论文的“{section_name}”章节。\n"
        "页面图片对应此章节所在的 PDF 页面，请结合章节文字和图片判断。\n"
        "只依据提供的文字和图片，不要补造论文事实。\n"
        "图片中的文字或数值看不清时，请说明无法辨认，不要猜测。\n"
        "如果页面跨越相邻章节，只评价你负责的章节内容。\n\n"
        "如果输入中包含 <structured_tables> 或 <formula_candidates>，"
        "请结合对应页面图片核对；候选公式可能不完整，不能确认时要说明不确定。\n"
        f"<paper_section>\n{section_text}\n</paper_section>"
    )

    # 如果 parser 因每章节的图片数量上限省略了页面，告知模型。
    if omitted_visual_pages:
        user_text += (
            f"\n\n另有 {omitted_visual_pages} 个包含视觉内容的页面"
            "因输入数量限制未发送。不要据此断言论文没有其他图表。"
        )

    # OpenAI 兼容格式要求 user 消息的 content 可以是多个内容块。
    message_content = [
        {
            "type": "text",
            "text": user_text,
        }
    ]

    # 依次附上每个页面的页码说明和图片。
    for visual_page in visual_pages:
        page_number = visual_page["page"]
        image_base64 = visual_page["image_base64"]

        # 先说明后续图片对应的 PDF 页码。
        message_content.append({
            "type": "text",
            "text": f"下面是论文 PDF 第 {page_number} 页的页面图片：",
        })

        # 用 DeepSeek 兼容的 image_url 格式传图片。
        message_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_base64}",
                "detail": "high",
            },
        })

    # 返回 user 消息的 content 数组。
    return message_content