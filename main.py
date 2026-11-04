import logging
from pathlib import Path
from typing import Any
# 捕获 pypdf 读取损坏 PDF 时常见的解析异常。
from pypdf.errors import PdfReadError

# 用于捕获 PyMuPDF 打开无效 PDF 时的文件数据异常。
import pymupdf

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException

from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.staticfiles import StaticFiles

from review_graph import build_review_graph
from pdf_parser import parse_and_group_pdf

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
MAX_PDF_SIZE = 30 * 1024 * 1024

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s",
)

app = FastAPI(title="Thesis Review System")

# 挂载 static 目录，浏览器通过 /static/... 访问前端文件
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# 返回前端首页
@app.get("/")
def root():
    return FileResponse(STATIC_DIR/"index.html")

#定义数据模型
class ApiResponse(BaseModel):
    code: int
    message:str
    data:Any=None


# 模块加载时创建并编译一次总图
review_graph = build_review_graph()

@app.post("/api/review", response_model=ApiResponse)
async def review(file: UploadFile=File(...))->ApiResponse:
    """接收 PDF、解析论文文本、调用 LangGraph 审稿流程、返回 JSON 报告"""

    if not file.filename or not file .filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400,detail="请上传pdf文件")

    pdf_bytes = await file.read(MAX_PDF_SIZE+1)

    if len(pdf_bytes) > MAX_PDF_SIZE:
        raise HTTPException(status_code=413, detail="PDF 文件不能超过 30 MB")

    if not pdf_bytes.startswith(b"%PDF-"):
        raise HTTPException(status_code=400, detail="文件内容不是有效的 PDF")

    try:
        # parser 现在返回章节文字、页码、图片及省略数量。
        paper_data = parse_and_group_pdf(pdf_bytes)
    except (ValueError,PdfReadError,pymupdf.FileDataError) as e:
        # PDF 无法读取或没有文字时，转换成 HTTP 400 错误。
        raise HTTPException(status_code=400, detail=f"PDF 无法解析：{e}",) from e
    except Exception as exc:
        # 记录完整错误堆栈供开发者排查。
        # 不把内部异常细节直接返回给前端。
        logger.exception("PDF 解析过程中发生未预期错误")

        raise HTTPException(
            status_code=500,
            detail="PDF 解析时发生内部错误，请稍后重试。",
        ) from exc
        # 将 parser 返回的四类数据放入 LangGraph 初始状态。
    result = review_graph.invoke({
        # 各职责对应的章节正文。
        "paper_sections": paper_data["sections"],

        # 各职责对应的 PDF 页码。
        "section_pages": paper_data["section_pages"],

        # 各职责对应的页面图片。
        "section_images": paper_data["section_images"],

        # 各职责对应的表格行列数据。
        "section_tables": paper_data["section_tables"],

        # 各职责对应的公式候选文字。
        "section_formulas": paper_data["section_formulas"],

        # 各职责因数量限制而省略的视觉页面数。
        "omitted_visual_pages": paper_data["omitted_visual_pages"],
    })

    # # 打印 parser 实际识别到的职责名称。
    # logger.info("识别到的职责：%s", list(paper_data["sections"].keys()))
    #
    # # 打印每个职责实际提取到的正文字符数。
    # logger.info(
    #     "各职责正文长度：%s",
    #     {
    #         role: len(text)
    #         for role, text in paper_data["sections"].items()
    #     },
    # )
    #
    # # 打印每个职责传给 agent 的页面图片数量。
    # logger.info(
    #     "各职责图片页数：%s",
    #     {
    #         role: len(images)
    #         for role, images in paper_data["section_images"].items()
    #     },
    # )
    # errors 是并行审稿节点返回的错误列表。
    errors=result.get("errors",[])
    if errors:
        # 把多个节点的错误合并成一条接口错误信息，方便前端显示。
        error_message="\n".join(errors)
        raise HTTPException(status_code=502, detail=error_message)


    return ApiResponse(
        code=200,
        message="审稿完成",
        data={
        "report": result["review_report"],
    },)



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


#微调 docker部署



