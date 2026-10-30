import logging
from pathlib import Path
from typing import Any
from urllib.request import Request

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException

from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.datastructures import UploadFile
from starlette.staticfiles import StaticFiles

from pdf_parser import parse_and_group_pdf

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
        # 下一步：把 pdf_bytes 交给 PDF 解析函数，再调用 LangGraph。

    paper_sections  = parse_and_group_pdf(pdf_bytes)



    return ApiResponse(
        code=200,
        message="PDF 接收成功",
        data={"filename": file.filename, "size": len(pdf_bytes)})











if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

