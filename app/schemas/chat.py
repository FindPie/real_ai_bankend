from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class Message(BaseModel):
    """消息模型"""

    role: Literal["user", "assistant", "system"] = Field(
        ..., description="消息角色: user/assistant/system"
    )
    content: str = Field(..., description="消息内容")


class ChatRequest(BaseModel):
    """聊天请求模型"""

    messages: List[Message] = Field(..., description="消息历史")
    model: str = Field(
        default="google/gemini-2.5-flash-lite", description="模型 ID"
    )
    stream: bool = Field(default=False, description="是否流式返回")
    web_search: bool = Field(default=False, description="是否启用联网搜索")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "messages": [{"role": "user", "content": "你好"}],
                    "model": "google/gemini-2.5-flash-lite",
                    "stream": False,
                }
            ]
        }
    }


class ChatResponse(BaseModel):
    """聊天响应模型"""

    content: str = Field(..., description="AI 回复内容")
    model: str = Field(..., description="使用的模型")
    images: Optional[List[str]] = Field(default=None, description="生成的图片列表")
    usage: Optional[dict] = Field(default=None, description="Token 使用情况")
