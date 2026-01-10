from typing import List, Literal

from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    """模型信息"""

    id: str = Field(..., description="模型 ID")
    name: str = Field(..., description="模型显示名称")
    provider: str = Field(..., description="提供商")
    type: Literal["chat", "vision", "image-gen"] = Field(
        ..., description="模型类型: chat/vision/image-gen"
    )


class ModelListResponse(BaseModel):
    """模型列表响应"""

    models: List[ModelInfo] = Field(..., description="可用模型列表")
    total: int = Field(..., description="模型总数")
