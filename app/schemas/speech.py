from typing import Literal, Optional

from pydantic import BaseModel, Field


class SpeechToTextRequest(BaseModel):
    """语音转文字请求模型 (Base64 方式)"""

    audio_data: str = Field(..., description="Base64 编码的音频数据")
    audio_format: Literal["pcm", "wav", "mp3", "m4a", "webm", "ogg"] = Field(
        default="wav", description="音频格式"
    )
    sample_rate: int = Field(default=16000, description="采样率 (Hz)")
    language: str = Field(default="zh-CN", description="语言代码")


class SpeechToTextResponse(BaseModel):
    """语音转文字响应模型"""

    text: str = Field(..., description="识别的文字内容")
    duration: Optional[float] = Field(default=None, description="音频时长 (秒)")
    confidence: Optional[float] = Field(default=None, description="置信度 (0-1)")


class TextToSpeechRequest(BaseModel):
    """文字转语音请求模型"""

    text: str = Field(..., description="要转换的文字", max_length=5000)
    voice: str = Field(default="xiaoyun", description="发音人")
    speed: int = Field(default=0, ge=-500, le=500, description="语速 (-500 到 500)")
    volume: int = Field(default=50, ge=0, le=100, description="音量 (0-100)")
    audio_format: Literal["pcm", "wav", "mp3"] = Field(
        default="mp3", description="输出音频格式"
    )


class TextToSpeechResponse(BaseModel):
    """文字转语音响应模型"""

    audio_data: str = Field(..., description="Base64 编码的音频数据")
    audio_format: str = Field(..., description="音频格式")
    duration: Optional[float] = Field(default=None, description="音频时长 (秒)")
