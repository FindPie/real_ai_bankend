from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat_service import chat_service

router = APIRouter()


@router.post(
    "/completions",
    response_model=ChatResponse,
    summary="发送聊天消息",
    description="向 AI 模型发送消息并获取回复",
)
async def chat_completions(request: ChatRequest) -> ChatResponse:
    """
    发送聊天消息

    - **messages**: 消息历史列表
    - **model**: 使用的模型 ID
    - **stream**: 是否流式返回 (此接口不支持，请使用 /stream 接口)
    - **web_search**: 是否启用联网搜索
    """
    if request.stream:
        raise HTTPException(
            status_code=400,
            detail="此接口不支持流式返回，请使用 /chat/stream 接口",
        )

    try:
        response = await chat_service.send_message(
            messages=request.messages,
            model=request.model,
            web_search=request.web_search,
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/stream",
    summary="流式聊天",
    description="向 AI 模型发送消息并以流式方式获取回复",
)
async def chat_stream(request: ChatRequest):
    """
    流式聊天

    返回 Server-Sent Events (SSE) 格式的流式响应
    """

    async def generate():
        try:
            async for chunk in chat_service.send_message_stream(
                messages=request.messages,
                model=request.model,
                web_search=request.web_search,
            ):
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
