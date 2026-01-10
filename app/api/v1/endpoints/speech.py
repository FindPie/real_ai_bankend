import base64
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from loguru import logger

from app.schemas.speech import (
    SpeechToTextRequest,
    SpeechToTextResponse,
    TextToSpeechRequest,
    TextToSpeechResponse,
)
from app.services.speech_service import speech_service

router = APIRouter()

# 支持的音频格式
SUPPORTED_AUDIO_FORMATS = {"pcm", "wav", "mp3", "m4a", "webm", "ogg", "flac", "amr"}
MAX_AUDIO_SIZE = 10 * 1024 * 1024  # 10MB


@router.post(
    "/recognize",
    response_model=SpeechToTextResponse,
    summary="语音识别 (Base64)",
    description="将 Base64 编码的音频数据转换为文字",
)
async def recognize_speech(request: SpeechToTextRequest) -> SpeechToTextResponse:
    """
    语音识别 - Base64 方式

    接收 Base64 编码的音频数据，返回识别的文字

    - **audio_data**: Base64 编码的音频数据
    - **audio_format**: 音频格式 (pcm, wav, mp3, m4a, webm, ogg)
    - **sample_rate**: 采样率，默认 16000
    - **language**: 语言代码，默认 zh-CN
    """
    try:
        # 验证 Base64 数据
        try:
            audio_bytes = base64.b64decode(request.audio_data)
        except Exception:
            raise HTTPException(status_code=400, detail="无效的 Base64 音频数据")

        # 检查音频大小
        if len(audio_bytes) > MAX_AUDIO_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"音频文件过大，最大支持 {MAX_AUDIO_SIZE // 1024 // 1024}MB",
            )

        result = await speech_service.speech_to_text(
            audio_data=request.audio_data,
            audio_format=request.audio_format,
            sample_rate=request.sample_rate,
            language=request.language,
        )
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"语音识别失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/recognize/upload",
    response_model=SpeechToTextResponse,
    summary="语音识别 (文件上传)",
    description="上传音频文件进行语音识别",
)
async def recognize_speech_upload(
    file: UploadFile = File(..., description="音频文件"),
    sample_rate: int = Form(default=16000, description="采样率"),
    language: str = Form(default="zh-CN", description="语言代码"),
) -> SpeechToTextResponse:
    """
    语音识别 - 文件上传方式

    直接上传音频文件进行识别

    支持的格式: pcm, wav, mp3, m4a, webm, ogg, flac, amr
    文件大小限制: 10MB
    """
    # 检查文件名
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    # 获取文件扩展名
    extension = file.filename.split(".")[-1].lower() if "." in file.filename else ""

    if extension not in SUPPORTED_AUDIO_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的音频格式: {extension}。支持: {', '.join(SUPPORTED_AUDIO_FORMATS)}",
        )

    # 读取文件内容
    content = await file.read()

    if len(content) > MAX_AUDIO_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"音频文件过大，最大支持 {MAX_AUDIO_SIZE // 1024 // 1024}MB",
        )

    # 转换为 Base64
    audio_base64 = base64.b64encode(content).decode("utf-8")

    try:
        result = await speech_service.speech_to_text(
            audio_data=audio_base64,
            audio_format=extension,
            sample_rate=sample_rate,
            language=language,
        )
        return result

    except Exception as e:
        logger.error(f"语音识别失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.websocket("/recognize/stream")
async def recognize_speech_stream(websocket: WebSocket):
    """
    实时语音识别 - WebSocket 流式接口

    用于接收麦克风实时音频流，实时返回识别结果

    协议:
    1. 客户端发送 JSON 配置: {"action": "start", "format": "pcm", "sample_rate": 16000}
    2. 客户端发送二进制音频数据 (持续)
    3. 服务端返回识别结果: {"text": "识别的文字", "is_final": false}
    4. 客户端发送: {"action": "stop"} 结束识别
    """
    await websocket.accept()
    logger.info("WebSocket 连接已建立")

    audio_buffer = bytearray()
    audio_format = "pcm"
    sample_rate = 16000

    try:
        while True:
            data = await websocket.receive()

            # 处理文本消息 (控制命令)
            if "text" in data:
                import json
                message = json.loads(data["text"])
                action = message.get("action")

                if action == "start":
                    audio_format = message.get("format", "pcm")
                    sample_rate = message.get("sample_rate", 16000)
                    audio_buffer.clear()
                    logger.info(f"开始录音: format={audio_format}, sample_rate={sample_rate}")
                    await websocket.send_json({"status": "started"})

                elif action == "stop":
                    # 处理累积的音频数据
                    if audio_buffer:
                        audio_base64 = base64.b64encode(bytes(audio_buffer)).decode("utf-8")
                        try:
                            result = await speech_service.speech_to_text(
                                audio_data=audio_base64,
                                audio_format=audio_format,
                                sample_rate=sample_rate,
                            )
                            await websocket.send_json({
                                "text": result.text,
                                "is_final": True,
                                "confidence": result.confidence,
                            })
                        except Exception as e:
                            await websocket.send_json({"error": str(e)})

                    audio_buffer.clear()
                    logger.info("录音结束")
                    await websocket.send_json({"status": "stopped"})

            # 处理二进制数据 (音频流)
            elif "bytes" in data:
                audio_buffer.extend(data["bytes"])

                # 每收集一定量的数据就进行识别 (实时识别)
                # 这里可以根据需要调整阈值
                if len(audio_buffer) >= sample_rate * 2:  # 约 1 秒的数据
                    audio_base64 = base64.b64encode(bytes(audio_buffer)).decode("utf-8")
                    try:
                        result = await speech_service.speech_to_text(
                            audio_data=audio_base64,
                            audio_format=audio_format,
                            sample_rate=sample_rate,
                        )
                        await websocket.send_json({
                            "text": result.text,
                            "is_final": False,
                        })
                    except Exception as e:
                        logger.error(f"实时识别错误: {e}")

    except WebSocketDisconnect:
        logger.info("WebSocket 连接已断开")
    except Exception as e:
        logger.error(f"WebSocket 错误: {e}")
        await websocket.close(code=1000)


@router.post(
    "/synthesize",
    response_model=TextToSpeechResponse,
    summary="语音合成",
    description="将文字转换为语音",
)
async def synthesize_speech(request: TextToSpeechRequest) -> TextToSpeechResponse:
    """
    语音合成 - 文字转语音

    - **text**: 要转换的文字 (最长 5000 字)
    - **voice**: 发音人 (默认 xiaoyun)
    - **speed**: 语速 (-500 到 500)
    - **volume**: 音量 (0-100)
    - **audio_format**: 输出格式 (pcm, wav, mp3)
    """
    try:
        result = await speech_service.text_to_speech(
            text=request.text,
            voice=request.voice,
            speed=request.speed,
            volume=request.volume,
            audio_format=request.audio_format,
        )
        return result

    except Exception as e:
        logger.error(f"语音合成失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
