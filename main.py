import logging
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException

from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.staticfiles import StaticFiles

from pdf_parser import parse_and_group_pdf
from agents.problem_contribution_agent import  problem_contribution_agent
from agents.method_agent import method_agent
from agents.evaluation_agent import evaluation_agent
from agents.validity_agent import validity_agent
from agents.conclusion_agent import conclusion_agent
from agents.report_agent import report_agent

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
    # 下一步：把 pdf_bytes 交给 PDF 解析函数，返回对应职责需要的正文
    paper_sections  = parse_and_group_pdf(pdf_bytes)

    print("识别到的职责：", paper_sections.keys())
    for role, text in paper_sections.items():
        print(role, len(text), repr(text[:200]))

    problem_text = paper_sections.get("problem_contribution", "")
    method_text = paper_sections.get("method", "")
    evaluation_text = paper_sections.get("evaluation", "")
    validity_text = paper_sections.get("validity", "")
    conclusion_text = paper_sections.get("conclusion", "")
    other_text = paper_sections.get("other", "")
    #将正文交给对应的agent处理
    problem_review = problem_contribution_agent(problem_text)
    method_review = method_agent(method_text)
    evaluation_review = evaluation_agent(evaluation_text)
    validity_review = validity_agent(validity_text)
    conclusion_review=conclusion_agent(conclusion_text)
    #汇总报告
    review_report = report_agent(
        problem_review=problem_review,
        method_review=method_review,
        evaluation_review=evaluation_review,
        validity_review=validity_review,
        conclusion_review=conclusion_review,
    )

    return ApiResponse(
        code=200,
        message="审稿完成",
        data={"result":review_report},)





if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

