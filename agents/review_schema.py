"""输出结构"""

from pydantic import BaseModel, Field


class ReviewResult(BaseModel):
    """单个审稿 agent 的结构化审稿结果。"""

    score: int | None = Field(
        default=None,ge=1,le=5,description="1到5分；材料不足以判断时为 null"
    )
    score_reason: str = Field(
        description="评分依据；材料不足时说明缺少什么"
    )
    strengths: list[str] = Field(
        description="该审稿职责下发现的优点"
    )
    issues: list[str] = Field(
        description="该审稿职责下发现的问题"
    )
    suggestions: list[str] = Field(
        description="针对问题给出的修改建议"
    )
    insufficient_information: list[str] = Field(
        description="当前输入中缺失、因此无法判断的信息"
    )