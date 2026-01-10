from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

from app.schemas.model import ModelInfo, ModelListResponse
from app.services.model_service import model_service

router = APIRouter()


@router.get(
    "",
    response_model=ModelListResponse,
    summary="获取所有可用模型",
    description="获取所有可用的 AI 模型列表",
)
async def get_models(
    provider: Optional[str] = Query(None, description="按提供商筛选"),
    type: Optional[str] = Query(None, description="按类型筛选: chat/vision/image-gen"),
) -> ModelListResponse:
    """
    获取可用模型列表

    可选筛选条件:
    - **provider**: 按提供商筛选 (Google, OpenAI, Anthropic, Alibaba, DeepSeek)
    - **type**: 按类型筛选 (chat, vision, image-gen)
    """
    if provider:
        models = model_service.get_models_by_provider(provider)
    elif type:
        models = model_service.get_models_by_type(type)
    else:
        return model_service.get_all_models()

    return ModelListResponse(models=models, total=len(models))


@router.get(
    "/{model_id:path}",
    response_model=ModelInfo,
    summary="获取模型详情",
    description="根据模型 ID 获取模型详细信息",
)
async def get_model(model_id: str) -> ModelInfo:
    """
    获取模型详情

    - **model_id**: 模型 ID (例如: google/gemini-2.5-flash-lite)
    """
    model = model_service.get_model_by_id(model_id)
    if not model:
        raise HTTPException(status_code=404, detail=f"模型不存在: {model_id}")
    return model
