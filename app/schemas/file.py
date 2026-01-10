from typing import Literal

from pydantic import BaseModel, Field


class FileParseRequest(BaseModel):
    """文件解析请求模型"""

    filename: str = Field(..., description="文件名")
    content_base64: str = Field(..., description="Base64 编码的文件内容")


class FileParseResponse(BaseModel):
    """文件解析响应模型"""

    content: str = Field(..., description="解析后的文本内容")
    file_type: Literal["PDF", "Word", "文本"] = Field(..., description="文件类型")
    filename: str = Field(..., description="原始文件名")
    size: int = Field(..., description="文件大小 (字节)")
